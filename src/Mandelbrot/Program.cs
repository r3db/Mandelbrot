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
                ViewportWidth  = 6 * 1920,
                ViewportHeight = 6 * 960,
                XMin = -2.2f,
                XMax = +1.0f,
                YMin = -1.0f,
                YMax = +1.0f,
            };

            Measure(() => Mandelbrot.RenderCpu1(bounds), "mandelbrot.cpu.1.png", "CPU: (Mandelbrot) Using Native GDI+ Bitmap!");
            Measure(() => Mandelbrot.RenderCpu2(bounds), "mandelbrot.cpu.2.png", "CPU: (Mandelbrot) Using byte Array!");
            Measure(() => Mandelbrot.RenderGpu1(bounds), "mandelbrot.gpu.1.png", "GPU: (Mandelbrot) Using byte Array!");
            Measure(() => Mandelbrot.RenderGpu2(bounds), "mandelbrot.gpu.2.png", "GPU: (Mandelbrot) Allocating Memory on GPU only!");
            Measure(() => Mandelbrot.RenderGpu3(bounds), "mandelbrot.gpu.3.png", "GPU: (Mandelbrot) Parallel.For!");
            //Measure(() => Mandelbrot.RenderGpu4(bounds), "mandelbrot.gpu.4.png", "GPU: (Mandelbrot) Multi-Device Parallel.For!");

            Measure(() => Julia.RenderCpu1(bounds), "julia.cpu.1.png", "CPU: (Julia) Using Native GDI+ Bitmap!");
            Measure(() => Julia.RenderCpu2(bounds), "julia.cpu.2.png", "CPU: (Julia) Using byte Array!");
            Measure(() => Julia.RenderGpu1(bounds), "julia.gpu.1.png", "GPU: (Julia) Using byte Array!");
            Measure(() => Julia.RenderGpu2(bounds), "julia.gpu.2.png", "GPU: (Julia) Allocating Memory on GPU only!");
            Measure(() => Julia.RenderGpu3(bounds), "julia.gpu.3.png", "GPU: (Julia) Parallel.For!");
            //Measure(() => Julia.RenderGpu4(bounds), "julia.gpu.4.png", "GPU: (Julia) Multi-Device Parallel.For!");

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