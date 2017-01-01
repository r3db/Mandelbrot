using System;
using System.Diagnostics;
using System.Drawing.Imaging;

namespace Mandelbrot
{
    internal static class Program
    {
        private static void Main()
        {
            const string path = @"test.cpu.png";
            
            var bounds = new Bounds
            {
                ViewportWidth = 1920,
                ViewporHeight = 960,
                XMin = -2.2f,
                XMax = +1.0f,
                YMin = -1.0f,
                YMax = +1.0f,
            };

            var sw = Stopwatch.StartNew();
            var bmp = Mandelbrot.RenderCpu(bounds);
            Console.WriteLine("{0}s", sw.Elapsed.TotalSeconds);

            bmp.Save(path, ImageFormat.Png);
            Console.ReadLine();
        }
    }
}