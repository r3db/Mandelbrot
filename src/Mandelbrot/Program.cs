using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;

namespace Mandelbrot
{
    internal static class Program
    {
        private static void Main()
        {
            const int scale  = 4;
            const int width  = scale * 1920;
            const int height = scale * 960;

            var bounds1 = new Bounds
            {
                ViewportWidth  = width,
                ViewportHeight = height,
                XMin = -2.2f,
                XMax = +1.0f,
                YMin = -1.0f,
                YMax = +1.0f,
            };

            var bounds2 = new Bounds
            {
                ViewportWidth  = width,
                ViewportHeight = height,
                XMin = -1.0f,
                XMax = +1.0f,
                YMin = -1.0f,
                YMax = +1.0f,
            };

            Measure(() => Mandelbrot.RenderCpu1(bounds1), "mandelbrot.cpu.1.png", false, "CPU: (Mandelbrot) Using Native GDI+ Bitmap!");
            Measure(() => Mandelbrot.RenderCpu2(bounds1), "mandelbrot.cpu.2.png", false, "CPU: (Mandelbrot) Using byte Array!");
            Measure(() => Mandelbrot.RenderGpu1(bounds1), "mandelbrot.gpu.1.png", true, "GPU: (Mandelbrot) Using byte Array!");
            Measure(() => Mandelbrot.RenderGpu2(bounds1), "mandelbrot.gpu.2.png", true, "GPU: (Mandelbrot) Allocating Memory on GPU only!");
            Measure(() => Mandelbrot.RenderGpu3(bounds1), "mandelbrot.gpu.3.png", true, "GPU: (Mandelbrot) Parallel.For!");

            Measure(() => Julia.RenderCpu1(bounds2), "julia.cpu.1.png", false, "CPU: (Julia) Using Native GDI+ Bitmap!");
            Measure(() => Julia.RenderCpu2(bounds2), "julia.cpu.2.png", false, "CPU: (Julia) Using byte Array!");
            Measure(() => Julia.RenderGpu1(bounds2), "julia.gpu.1.png", true, "GPU: (Julia) Using byte Array!");
            Measure(() => Julia.RenderGpu2(bounds2), "julia.gpu.2.png", true, "GPU: (Julia) Allocating Memory on GPU only!");
            Measure(() => Julia.RenderGpu3(bounds2), "julia.gpu.3.png", true, "GPU: (Julia) Parallel.For!");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<Image> func, string fileName, bool isGpu, string description)
        {
            const string format = "{0,9}";

            Func<Stopwatch, string> formatElapsedTime = w => w.Elapsed.TotalSeconds >= 1
                ? string.Format(CultureInfo.InvariantCulture, format + "  (s)", w.Elapsed.TotalSeconds)
                : w.Elapsed.TotalMilliseconds >= 1
                    ? string.Format(CultureInfo.InvariantCulture, format + " (ms)", w.Elapsed.TotalMilliseconds)
                    : string.Format(CultureInfo.InvariantCulture, format + " (μs)", w.Elapsed.TotalMilliseconds * 1000);

            Action consoleColor = () =>
            {
                Console.ForegroundColor = isGpu
                    ? ConsoleColor.White
                    : ConsoleColor.Cyan;
            };

            var sw1 = Stopwatch.StartNew();
            var result1 = func();
            sw1.Stop();

            Func<Stopwatch, string> bandwidth = w => string.Format(CultureInfo.InvariantCulture, "{0,8:F4} GB/s", (result1.Width * result1.Height * 3) / (w.Elapsed.TotalMilliseconds * 1000000));

            Console.WriteLine(new string('-', 49));
            Console.WriteLine(description);
            consoleColor();
            Console.WriteLine("{0,9} - {1} - {2} [Cold]", result1, formatElapsedTime(sw1), bandwidth(sw1));
            Console.ResetColor();
            result1.Save(fileName, ImageFormat.Png);

            var sw2 = Stopwatch.StartNew();
            var result2 = func();
            sw2.Stop();
            consoleColor();
            Console.WriteLine("{0,9} - {1} - {2} [Warm]", result2, formatElapsedTime(sw2), bandwidth(sw2));
            Console.ResetColor();
        }
    }
}