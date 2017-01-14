using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace Mandelbrot
{
    internal static class BitmapUtility
    {
        private const PixelFormat PixelFormat = System.Drawing.Imaging.PixelFormat.Format24bppRgb;

        internal static Bitmap FromByteArray(byte[] data, int width, int height)
        {
            var pinnedData = GCHandle.Alloc(data, GCHandleType.Pinned);
            var result = new Bitmap(width, height, 3 * width * sizeof(byte), PixelFormat, pinnedData.AddrOfPinnedObject());

            pinnedData.Free();
            return result;
        }
    }
}