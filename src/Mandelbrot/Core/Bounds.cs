using System;

namespace Mandelbrot
{
    internal struct Bounds
    {
        // Todo: Should we promote these to Properties!?
        public int ViewportWidth;
        public int ViewportHeight;

        public float XMin;
        public float XMax;
        public float YMin;
        public float YMax;

        public void AdjustAspectRatio()
        {
            var vr = ViewportHeight / (float)ViewportWidth;
            var gr = (YMax - YMin) / (XMax - XMin);

            var xd = gr > vr
                ? (ViewportWidth * ((YMax - YMin) / ViewportHeight) - (XMax - XMin)) * 0.5f
                : 0f;

            var yd = gr > vr
                ? 0f
                : (ViewportHeight * ((XMax - XMin) / ViewportWidth) - (YMax - YMin)) * 0.5f;

            XMin -= xd;
            XMax += xd;
            YMin -= yd;
            YMax += yd;
        }
    }
}