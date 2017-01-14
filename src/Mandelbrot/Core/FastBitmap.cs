using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace Mandelbrot
{
    // Ignore the fact that we've not implemented IDisposable
    internal sealed class FastBitmap
    {
        private const PixelFormat PixelFormat = System.Drawing.Imaging.PixelFormat.Format24bppRgb;

        private readonly Bitmap _bitmap;
        private readonly BitmapData _data;

        internal FastBitmap(int width, int height)
        {
            _bitmap = new Bitmap(width, height, PixelFormat);
            _data = _bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, PixelFormat);
        }

        // Todo: Fix!
        //~FastBitmap()
        //{
        //    _bitmap.UnlockBits(_data);
        //}

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