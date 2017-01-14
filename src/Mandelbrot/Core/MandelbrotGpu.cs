using System;
using System.Drawing;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace Mandelbrot
{
    // Todo: Use Fixed Blocks!
    internal static class MandelbrotGpu
    {
        // Alea Parallel Linq!
        internal static Image RenderGpu1(Bounds bounds)
        {
            bounds.AdjustAspectRatio();

            var width = bounds.ViewportWidth;
            var height = bounds.ViewportHeight;
            var scale = (bounds.XMax - bounds.XMin) / width;

            var resultLength = 3 * width * height;
            var resultMemory = Gpu.Default.AllocateDevice<byte>(resultLength);
            var resultDevPtr = new deviceptr<byte>(resultMemory.Handle);

            Gpu.Default.For(0, width * height, i =>
            {
                var x = i % width;
                var y = i / width;
                var offset = 3 * i;

                // ReSharper disable once PossibleLossOfFraction
                var c = new Complex
                {
                    Real      = bounds.XMin + x * scale,
                    Imaginary = bounds.YMin + y * scale,
                };

                ComputeMandelbrotAtOffset(resultDevPtr, c, offset);
            });

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }

        // Custom!
        internal static Image RenderGpu2(Bounds bounds)
        {
            bounds.AdjustAspectRatio();

            var width = bounds.ViewportWidth;
            var height = bounds.ViewportHeight;
            var scale = (bounds.XMax - bounds.XMin) / width;

            var resultLength = 3 * width * height;
            var resultMemory = Gpu.Default.AllocateDevice<byte>(resultLength);
            var resultDevPtr = new deviceptr<byte>(resultMemory.Handle);

            var lp = ComputeLaunchParameters(bounds);
            
            Gpu.Default.Launch(() =>
            {
                var x = blockDim.x * blockIdx.x + threadIdx.x;
                var y = blockDim.y * blockIdx.y + threadIdx.y;
                var offset = 3 * (y * width + x);

                if (offset < resultLength)
                {
                    var c = new Complex
                    {
                        Real      = bounds.XMin + x * scale,
                        Imaginary = bounds.YMin + y * scale,
                    };

                    ComputeMandelbrotAtOffset(resultDevPtr, c, offset);
                }
            }, lp);

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }
        
        // Helpers!
        private static void ComputeMandelbrotAtOffset(deviceptr<byte> result, Complex c, int offset)
        {
            var z = c;

            for (byte i = 0; i < 255; ++i)
            {
                z = z * z + c;

                if (z.Magnitude() >= 2)
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
            var width = bounds.ViewportWidth;
            var height = bounds.ViewportHeight;
            return new LaunchParam(new dim3((width + (tpb - 1)) / tpb, (height + (tpb - 1)) / tpb), new dim3(tpb, tpb));
        }
    }
}