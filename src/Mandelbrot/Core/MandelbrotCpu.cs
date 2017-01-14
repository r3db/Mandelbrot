using System;
using System.Drawing;
using System.Threading.Tasks;

namespace Mandelbrot
{
    internal static class MandelbrotCpu
    {
        // Native GDI+ Bitmap!
        internal static Image Render1(Bounds bounds)
        {
            bounds.AdjustAspectRatio();

            var width  = bounds.ViewportWidth;
            var height = bounds.ViewportHeight;
            var scale = (bounds.XMax - bounds.XMin) / width;
            var result = new FastBitmap(width, height);

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var c = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);

                    // ReSharper disable once AccessToModifiedClosure
                    ComputeMandelbrotAtOffset(c, i => result.SetPixel(x, y, Color.FromArgb(i, i, i)));
                }
            });

            return result.Bitmap;
        }

        // Byte Array!
        internal static Image Render2(Bounds bounds)
        {
            bounds.AdjustAspectRatio();

            var width = bounds.ViewportWidth;
            var height = bounds.ViewportHeight;
            var scale = (bounds.XMax - bounds.XMin) / width;
            var result = new byte[3 * width * height];

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var c = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);

                    ComputeMandelbrotAtOffset(c, i =>
                    {
                        // ReSharper disable once AccessToModifiedClosure
                        var offset = 3 * (y * width + x);

                        result[offset + 0] = i;
                        result[offset + 1] = i;
                        result[offset + 2] = i;
                    });
                }
            });

            return BitmapUtility.FromByteArray(result, width, height);
        }

        // Helpers!
        private static void ComputeMandelbrotAtOffset(Complex c, Action<byte> action)
        {
            var z = c;

            for (byte i = 0; i < 255; ++i)
            {
                z = z * z + c;

                if (z.Magnitude() >= 2)
                {
                    action(i);
                    break;
                }
            }
        }
    }
}