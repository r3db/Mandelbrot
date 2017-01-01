using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;

namespace Mandel
{
    internal static class Mandelbrot
    {
        internal static Bitmap RenderCpu(Bounds bounds)
        {
            var result = new FastBitmap(bounds.ViewportWidth, bounds.ViewporHeight);
            var palette = AvailableColors();

            bounds.AdjustAspectRatio(); ;

            // At this point after 'AdjustAspectRatio' is safe to assume there's only one scale on both x an y!
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Parallel.For(0, bounds.ViewporHeight, y =>
            {
                for (var x = 0; x < bounds.ViewportWidth; ++x)
                {
                    var c = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);
                    // Note: The following assigment is actually a copy!
                    var z = c;

                    foreach (var item in palette)
                    {
                        z = (z * z) + c;

                        if (z.Magnitude() >= 2)
                        {
                            result.SetPixel(x, y, item);
                            break;
                        }
                    }
                }
            });

            return result.Bitmap;
        }

        private static IList<Color> AvailableColors()
        {
            var result = new List<Color>();

            for (var i = 0; i <= byte.MaxValue; ++i)
            {
                result.Add(Color.FromArgb(255, i, i, i));
            }

            return result;
        }
    }
}