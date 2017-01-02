using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Mandelbrot
{
    // Ignore the fact that we've not implemented IDisposable
    internal sealed class FastBitmap
    {
        private readonly Bitmap _bitmap;
        private readonly BitmapData _data;

        internal FastBitmap(int width, int height)
        {
            _bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            _data = Bitmap.LockBits(new Rectangle(0, 0, Bitmap.Width, Bitmap.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
        }

        ~FastBitmap()
        {
            Bitmap.UnlockBits(_data);
        }

        internal static FastBitmap FromByteArray(byte[] data, int width, int height)
        {
            var result = new FastBitmap(width, height);

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var offset = 3 * (y * width + x);

                    result.SetPixel(x, y, Color.FromArgb(
                        data[offset + 0],
                        data[offset + 1],
                        data[offset + 2]));
                }
            });

            return result;
        }

        internal unsafe void SetPixel(int x, int y, Color color)
        {
            var pixel = (byte*)_data.Scan0.ToPointer() + (y * _data.Stride + 3 * x);

            pixel[0] = color.B;
            pixel[1] = color.G;
            pixel[2] = color.R;
        }

        internal Bitmap Bitmap => _bitmap;
    }
}