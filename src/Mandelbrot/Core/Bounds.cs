using System;

namespace Mandelbrot
{
    internal struct Bounds
    {
        // Todo: Should we promote these to Properties!?
        public int Width;
        public int Height;

        public float XMin;
        public float XMax;
        public float YMin;
        public float YMax;

        public void AdjustAspectRatio()
        {
            var vr = Height / (float)Width;
            var gr = (YMax - YMin) / (XMax - XMin);

            var xd = gr > vr
                ? (Width * ((YMax - YMin) / Height) - (XMax - XMin)) * 0.5f
                : 0f;

            var yd = gr > vr
                ? 0f
                : (Height * ((XMax - XMin) / Width) - (YMax - YMin)) * 0.5f;

            XMin -= xd;
            XMax += xd;
            YMin -= yd;
            YMax += yd;
        }
    }
}