using System;
using System.Drawing;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace Mandelbrot
{
    internal static class MandelbrotGpu
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
                    // ReSharper disable once PossibleLossOfFraction
                    var c = new Complex
                    {
                        Real      = bounds.XMin + x * scale,
                        Imaginary = bounds.YMin + y * scale,
                    };

                    ComputeMandelbrotAtOffset(resultDevPtr, c, offset);
                }
            });

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }

        // Custom!
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
                var i = blockDim.x * blockIdx.x + threadIdx.x;
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

                    ComputeMandelbrotAtOffset(resultDevPtr, c, offset);
                }
            }, lp);

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }

        // Fixed Block Size!
        internal static Image Render3(Bounds bounds)
        {
            bounds.AdjustAspectRatio();

            var width = bounds.Width;
            var height = bounds.Height;
            var scale = (bounds.XMax - bounds.XMin) / width;

            var resultLength = ColorComponents * width * height;
            var resultMemory = Gpu.Default.AllocateDevice<byte>(resultLength);
            var resultDevPtr = new deviceptr<byte>(resultMemory.Handle);

            var lp = new LaunchParam(256, 256);

            Gpu.Default.Launch(() =>
            {
                var i = blockDim.x * blockIdx.x + threadIdx.x;

                while (ColorComponents * i < resultLength)
                {
                    var x = i % width;
                    var y = i / width;
                    var offset = ColorComponents * i;

                    var c = new Complex
                    {
                        Real      = bounds.XMin + x * scale,
                        Imaginary = bounds.YMin + y * scale,
                    };

                    ComputeMandelbrotAtOffset(resultDevPtr, c, offset);

                    i += gridDim.x * blockDim.x;
                }
            }, lp);

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(resultMemory), width, height);
        }
        
        // Helpers!
        private static void ComputeMandelbrotAtOffset(deviceptr<byte> result, Complex c, int offset)
        {
            var z = c;

            for (byte i = 0; i < byte.MaxValue; ++i)
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
            const int tpb = 256;
            return new LaunchParam((ColorComponents * bounds.Width * bounds.Height + (tpb - 1)) / tpb, tpb);
        }
    }
}