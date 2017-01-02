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
        internal static Image RenderCpu1(Bounds bounds)
        {
            var result = new FastBitmap(bounds.ViewportWidth, bounds.ViewportHeight);
            bounds.AdjustAspectRatio();

            // At this point after 'AdjustAspectRatio' is safe to assume there's only one scale on both x an y!
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Parallel.For(0, bounds.ViewportHeight, y =>
            {
                for (var x = 0; x < bounds.ViewportWidth; ++x)
                {
                    var c = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);
                    // Note: The following assigment is actually a copy!
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

        internal static Image RenderCpu2(Bounds bounds)
        {
            var result = new byte[bounds.ViewportWidth * bounds.ViewportHeight * 3];
            bounds.AdjustAspectRatio();

            // At this point after 'AdjustAspectRatio' is safe to assume there's only one scale on both x an y!
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Parallel.For(0, bounds.ViewportHeight, y =>
            {
                for (var x = 0; x < bounds.ViewportWidth; ++x)
                {
                    var offset = 3 * (y * bounds.ViewportWidth + x);

                    var c = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);

                    Kernel(result, offset, c);
                }
            });

            return FastBitmap.FromByteArray(result, bounds.ViewportWidth, bounds.ViewportHeight).Bitmap;
        }

        internal static Image RenderGpu1(Bounds bounds)
        {
            var result = new byte[bounds.ViewportWidth * bounds.ViewportHeight * 3];

            bounds.AdjustAspectRatio();

            // At this point after 'AdjustAspectRatio' is safe to assume there's only one scale on both x an y!
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;
            var lp = new LaunchParam(new dim3(bounds.ViewportWidth, bounds.ViewportHeight), new dim3(1));

            Gpu.Default.Launch(() =>
            {
                var x = blockIdx.x;
                var y = blockIdx.y;
                var offset = 3 * (y * gridDim.x + x);

                var c = new Complex
                {
                    Real = bounds.XMin + x * scale,
                    Imaginary = bounds.YMin + y * scale,
                };

                Kernel(result, offset, c);
            }, lp);

            return FastBitmap.FromByteArray(result, bounds.ViewportWidth, bounds.ViewportHeight).Bitmap;
        }

        internal static Image RenderGpu2(Bounds bounds)
        {
            var result = new byte[bounds.ViewportWidth * bounds.ViewportHeight * 3];

            bounds.AdjustAspectRatio();

            // At this point after 'AdjustAspectRatio' is safe to assume there's only one scale on both x an y!
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;
            var lp = new LaunchParam(new dim3(bounds.ViewportWidth, bounds.ViewportHeight), new dim3(1));

            Gpu.Default.Launch(() =>
            {
                var x = blockIdx.x;
                var y = blockIdx.y;
                var offset = 3 * (y * gridDim.x + x);

                var c = new Complex
                {
                    Real = bounds.XMin + x * scale,
                    Imaginary = bounds.YMin + y * scale,
                };

                Kernel(result, offset, c);
            }, lp);

            return FastBitmap.FromByteArray(result, bounds.ViewportWidth, bounds.ViewportHeight).Bitmap;
        }

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void Kernel(byte[] result, int offset, Complex c)
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