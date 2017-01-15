using System;
using System.Drawing;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace Mandelbrot
{
    // Todo: Use Fixed Blocks!
    internal static class JuliaGpu
    {
        private const int ColorComponents = 3;

        // Alea Parallel.For!
        internal static Image Render1(Bounds bounds)
        {
            bounds.AdjustAspectRatio();

            var width = bounds.Width;
            var height = bounds.Height;
            var scale = (bounds.XMax - bounds.XMin) / width;

            var resultLength = ColorComponents * width * height;
            var resultMemory = Gpu.Default.AllocateDevice<byte>(resultLength);
            var resultDevPtr = new deviceptr<byte>(resultMemory.Handle);

            Gpu.Default.For(0, width * height, i =>
            {
                var x = i % width;
                var y = i / width;
                var offset = ColorComponents * i;

                if (offset < resultLength)
                {
                    var c = new Complex
                    {
                        Real      = bounds.XMin + x * scale,
                        Imaginary = bounds.YMin + y * scale,
                    };

                    ComputeJuliaSetAtOffset(resultDevPtr, c, offset);
                }                
            });

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }

        // GPU: Custom!
        internal static Image Render2(Bounds bounds)
        {
            bounds.AdjustAspectRatio();

            var width = bounds.Width;
            var height = bounds.Height;
            var scale = (bounds.XMax - bounds.XMin) / width;

            var resultLength = ColorComponents * width * height;
            var resultMemory = Gpu.Default.AllocateDevice<byte>(resultLength);
            var resultDevPtr = new deviceptr<byte>(resultMemory.Handle);

            var lp = ComputeLaunchParameters(bounds);

            Gpu.Default.Launch(() =>
            {
                var x = blockDim.x * blockIdx.x + threadIdx.x;
                var y = blockDim.y * blockIdx.y + threadIdx.y;
                var offset = ColorComponents * (y * bounds.Width + x);

                var c = new Complex
                {
                    Real      = bounds.XMin + x * scale,
                    Imaginary = bounds.YMin + y * scale,
                };

                ComputeJuliaSetAtOffset(resultDevPtr, c, offset);
            }, lp);

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void ComputeJuliaSetAtOffset(deviceptr<byte> result, Complex a, int offset)
        {
            var c = new Complex
            {
                Real       = -0.8f,
                Imaginary = +0.156f,
            };

            for (byte i = 0; i < 255; ++i)
            {
                a = a * a + c;

                if (a.Magnitude() >= 2)
                {
                    result[offset + 0] = i;
                    result[offset + 1] = i;
                    result[offset + 2] = i;
                    break;
                }
            }
        }

        private static LaunchParam ComputeLaunchParameters(Bounds bounds)
        {
            const int tpb = 32;
            var width = bounds.Width;
            var height = bounds.Height;
            return new LaunchParam(new dim3((width + (tpb - 1)) / tpb, (height + (tpb - 1)) / tpb), new dim3(tpb, tpb));
        }
    }
}