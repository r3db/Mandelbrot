using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace Mandelbrot
{
    internal static class Program
    {
        private static void Main()
        {
            var bounds = new Bounds
            {
                ViewportWidth  = 2 * 1920,
                ViewportHeight = 2 * 960,
                XMin = -2.2f,
                XMax = +1.0f,
                YMin = -1.0f,
                YMax = +1.0f,
            };

            Measure(() => Mandelbrot.RenderCpu1(bounds), "test.cpu.1.png", "CPU: Using Native GDI+ Bitmap!");
            Measure(() => Mandelbrot.RenderCpu2(bounds), "test.cpu.2.png", "CPU: Using byte Array!");
            Measure(() => Mandelbrot.RenderGpu1(bounds), "test.gpu.1.png", "GPU: Using byte Array!");
            Measure(() => Mandelbrot.RenderGpu2(bounds), "test.gpu.2.png", "GPU: Allocating Memory on GPU only!");
            Measure(() => Mandelbrot.RenderGpu3(bounds), "test.gpu.3.png", "GPU: Parallel.For!");
            Measure(() => Mandelbrot.RenderGpu4(bounds), "test.gpu.4.png", "GPU: Multi-Device Parallel.For!");

            Console.WriteLine("Done!");
            Console.ReadLine();
        }

        private static void Measure(Func<Image> func, string fileName, string description)
        {
            Func<Stopwatch, string> formatElapsedTime = (watch) => watch.Elapsed.TotalSeconds >= 1
                ? $"{watch.Elapsed.TotalSeconds}s"
                : $"{watch.ElapsedMilliseconds}ms";

            var sw1 = Stopwatch.StartNew();
            var bmp1 = func();
            sw1.Stop();
            Console.WriteLine(description);
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("{0} [Cold]", formatElapsedTime(sw1));
            Console.ResetColor();
            bmp1.Save(fileName, ImageFormat.Png);

            Console.WriteLine();

            var sw2 = Stopwatch.StartNew();
            func();
            sw2.Stop();
            Console.WriteLine(description);
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("{0} [Warm]", formatElapsedTime(sw2));
            Console.ResetColor();

            Console.WriteLine(new string('-', 40));
            Console.WriteLine();
        }
    }
}