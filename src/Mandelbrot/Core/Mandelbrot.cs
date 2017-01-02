using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using Alea;
using Alea.CSharp;

namespace Mandelbrot
{
    internal static class Mandelbrot
    {
        // CPU: Using Native GDI+ Bitmap!
        internal static Image RenderCpu1(Bounds bounds)
        {
            var result = new FastBitmap(bounds.ViewportWidth, bounds.ViewportHeight);
            bounds.AdjustAspectRatio();
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Parallel.For(0, bounds.ViewportHeight, y =>
            {
                for (var x = 0; x < bounds.ViewportWidth; ++x)
                {
                    var c = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);
                    var z = c;

                    for (byte i = 0; i < 255; ++i)
                    {
                        z = z * z + c;

                        if (z.Magnitude() >= 2)
                        {
                            result.SetPixel(x, y, Color.FromArgb(i, i, i));
                            break;
                        }
                    }
                }
            });

            return result.Bitmap;
        }

        // CPU: Using byte Array!
        internal static Image RenderCpu2(Bounds bounds)
        {
            var result = new byte[bounds.ViewportWidth * bounds.ViewportHeight * 3];
            bounds.AdjustAspectRatio();
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Parallel.For(0, bounds.ViewportHeight, y =>
            {
                for (var x = 0; x < bounds.ViewportWidth; ++x)
                {
                    var offset = 3 * (y * bounds.ViewportWidth + x);
                    var c = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);

                    ComputeMandelbrotAtOffset(result, c, offset);
                }
            });

            return FastBitmap.FromByteArray(result, bounds.ViewportWidth, bounds.ViewportHeight);
        }

        // GPU: Using byte Array!
        internal static Image RenderGpu1(Bounds bounds)
        {
            var result = new byte[bounds.ViewportWidth * bounds.ViewportHeight * 3];
            bounds.AdjustAspectRatio();
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;
            var lp = new LaunchParam(new dim3(bounds.ViewportWidth, bounds.ViewportHeight), new dim3(1));

            Gpu.Default.Launch(() =>
            {
                var x = blockIdx.x;
                var y = blockIdx.y;
                var offset = 3 * (y * gridDim.x + x);

                var c = new Complex
                {
                    Real      = bounds.XMin + x * scale,
                    Imaginary = bounds.YMin + y * scale,
                };

                ComputeMandelbrotAtOffset(result, c, offset);
            }, lp);

            return FastBitmap.FromByteArray(result, bounds.ViewportWidth, bounds.ViewportHeight);
        }

        // GPU: Allocating Memory on GPU only!
        internal static Image RenderGpu2(Bounds bounds)
        {
            var deviceResult = Gpu.Default.Allocate<byte>(bounds.ViewportWidth * bounds.ViewportHeight * 3);
            bounds.AdjustAspectRatio();
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;
            var lp = new LaunchParam(new dim3(bounds.ViewportWidth, bounds.ViewportHeight), new dim3(1));

            Gpu.Default.Launch(() =>
            {
                var x = blockIdx.x;
                var y = blockIdx.y;
                var offset = 3 * (y * gridDim.x + x);

                var c = new Complex
                {
                    Real      = bounds.XMin + x * scale,
                    Imaginary = bounds.YMin + y * scale,
                };

                ComputeMandelbrotAtOffset(deviceResult, c, offset);
            }, lp);

            return FastBitmap.FromByteArray(Gpu.CopyToHost(deviceResult), bounds.ViewportWidth, bounds.ViewportHeight);
        }

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void ComputeMandelbrotAtOffset(byte[] result, Complex c, int offset)
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
    }
}