using System;

namespace Mandelbrot
{
    internal struct Complex
    {
        // Todo: Should we promote these to Properties!?
        readonly float Real;
        readonly float Imaginary;

        public Complex(float real, float imaginary)
        {
            Real = real;
            Imaginary = imaginary;
        }

        public float Magnitude()
        {
            return Real * Real + Imaginary * Imaginary;
        }

        public static Complex operator *(Complex first, Complex second)
        {
            return new Complex(first.Real * second.Real - first.Imaginary * second.Imaginary, first.Imaginary * second.Real + first.Real * second.Imaginary);
        }

        public static Complex operator +(Complex first, Complex second)
        {
            return new Complex(first.Real + second.Real, first.Imaginary + second.Imaginary);
        }
    }
}