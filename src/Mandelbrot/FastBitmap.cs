using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace Mandel
{
    // Ignore the fact that we've not implemented IDisposable
    internal sealed class FastBitmap
    {
        private readonly BitmapData _data;
    
        public FastBitmap(int width, int height)
        {
            Bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            _data = Bitmap.LockBits(new Rectangle(0, 0, Bitmap.Width, Bitmap.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
        }

        ~FastBitmap()
        {
            Bitmap.UnlockBits(_data);
        }

        public unsafe void SetPixel(int x, int y, Color color)
        {
            var pixel = (byte*)_data.Scan0.ToPointer() + ((y * _data.Stride) + (3 * x));

            pixel[0] = color.B;
            pixel[1] = color.G;
            pixel[2] = color.R;
        }

        internal Bitmap Bitmap { get; }
    }
}