using System;
using System.Drawing;
using System.Threading.Tasks;

namespace Mandelbrot
{
    internal static class JuliaCpu
    {
        private const int ColorComponents = 3;

        // Native GDI+ Bitmap!
        internal static Image Render1(Bounds bounds)
        {
            bounds.AdjustAspectRatio();

            var width = bounds.Width;
            var height = bounds.Height;
            var scale = (bounds.XMax - bounds.XMin) / width;
            var result = new FastBitmap(width, height);

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var c = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);

                    // ReSharper disable once AccessToModifiedClosure
                    ComputeJuliaSetAtOffset(c, i => result.SetPixel(x, y, Color.FromArgb(i, i, i)));
                }
            });

            return result.Bitmap;
        }

        // Byte Array!
        internal static Image Render2(Bounds bounds)
        {
            bounds.AdjustAspectRatio();

            var width = bounds.Width;
            var height = bounds.Height;
            var scale = (bounds.XMax - bounds.XMin) / width;
            var result = new byte[ColorComponents * width * height];

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var c = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);

                    ComputeJuliaSetAtOffset(c, i =>
                    {
                        var offset = ColorComponents * (y * width + x);

                        result[offset + 0] = i;
                        result[offset + 1] = i;
                        result[offset + 2] = i;
                    });
                }
            });

            return BitmapUtility.FromByteArray(result, width, height);
        }

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void ComputeJuliaSetAtOffset(Complex a, Action<byte> action)
        {
            var c = new Complex
            {
                Real      = -0.8f,
                Imaginary = +0.156f,
            };

            for (byte i = 0; i < 255; ++i)
            {
                a = a * a + c;

                if (a.Magnitude() >= 2)
                {
                    action(i);
                    break;
                }
            }
        }
    }
}