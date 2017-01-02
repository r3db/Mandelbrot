using System;
using System.Diagnostics;
using System.Drawing.Imaging;

namespace Mandelbrot
{
    internal static class Program
    {
        private static void Main()
        {
            var bounds = new Bounds
            {
                ViewportWidth = 1920,
                ViewportHeight = 960,
                XMin = -2.2f,
                XMax = +1.0f,
                YMin = -1.0f,
                YMax = +1.0f,
            };

            var sw1 = Stopwatch.StartNew();
            var bmp1 = Mandelbrot.RenderCpu1(bounds);
            Console.WriteLine("{0}s", sw1.Elapsed.TotalSeconds);
            bmp1.Save(@"test.cpu.1.png", ImageFormat.Png);

            var sw2 = Stopwatch.StartNew();
            var bmp2 = Mandelbrot.RenderCpu2(bounds);
            Console.WriteLine("{0}s", sw2.Elapsed.TotalSeconds);
            bmp2.Save(@"test.cpu.2.png", ImageFormat.Png);

            var sw3 = Stopwatch.StartNew();
            var bmp3 = Mandelbrot.RenderGpu1(bounds);
            Console.WriteLine("{0}s", sw3.Elapsed.TotalSeconds);
            bmp3.Save(@"test.gpu.1.png", ImageFormat.Png);

            var sw4 = Stopwatch.StartNew();
            var bmp4 = Mandelbrot.RenderGpu1(bounds);
            Console.WriteLine("{0}s", sw4.Elapsed.TotalSeconds);
            bmp4.Save(@"test.gpu.2.png", ImageFormat.Png);

            var sw5 = Stopwatch.StartNew();
            var bmp5 = Mandelbrot.RenderGpu2(bounds);
            Console.WriteLine("{0}s", sw5.Elapsed.TotalSeconds);
            bmp5.Save(@"test.gpu.3.png", ImageFormat.Png);

            Console.WriteLine("Done!");
            Console.ReadLine();
        }
    }
}