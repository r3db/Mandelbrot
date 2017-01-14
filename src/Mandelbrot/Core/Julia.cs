using System;
using System.Drawing;
using System.Threading.Tasks;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace Mandelbrot
{
    internal static class Julia
    {
        // CPU: Using Native GDI+ Bitmap!
        internal static Image RenderCpu1(Bounds bounds)
        {
            var result = new FastBitmap(bounds.ViewportWidth, bounds.ViewportHeight);
            bounds.AdjustAspectRatio();
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Parallel.For(0, bounds.ViewportHeight, y =>
            {
                for (var x = 0; x < bounds.ViewportWidth; ++x)
                {
                    var c = new Complex(-0.8f, 0.156f);
                    var a = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);

                    for (byte i = 0; i < 255; ++i)
                    {
                        a = a * a + c;

                        if (a.Magnitude() >= 2)
                        {
                            result.SetPixel(x, y, Color.FromArgb(i, i, i));
                            break;
                        }
                    }
                }
            });

            return result.Bitmap;
        }

        // CPU: Using byte Array!
        internal static Image RenderCpu2(Bounds bounds)
        {
            var result = new byte[3 * bounds.ViewportWidth * bounds.ViewportHeight];
            bounds.AdjustAspectRatio();
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Parallel.For(0, bounds.ViewportHeight, y =>
            {
                for (var x = 0; x < bounds.ViewportWidth; ++x)
                {
                    var offset = 3 * (y * bounds.ViewportWidth + x);
                    var a = new Complex(bounds.XMin + x * scale, bounds.YMin + y * scale);

                    ComputeJuliaSetAtOffset(result, a, offset);
                }
            });

            return BitmapUtility.FromByteArray(result, bounds.ViewportWidth, bounds.ViewportHeight);
        }

        // GPU: Using byte Array!
        internal static Image RenderGpu1(Bounds bounds)
        {
            var result = new byte[3 * bounds.ViewportWidth * bounds.ViewportHeight];
            bounds.AdjustAspectRatio();
            var lp = ComputeLaunchParameters(bounds);
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Gpu.Default.Launch(() =>
            {
                var x = blockDim.x * blockIdx.x + threadIdx.x;
                var y = blockDim.y * blockIdx.y + threadIdx.y;
                var offset = 3 * (y * bounds.ViewportWidth + x);

                var a = new Complex
                {
                    Real      = bounds.XMin + x * scale,
                    Imaginary = bounds.YMin + y * scale,
                };

                ComputeJuliaSetAtOffset(result, a, offset);
            }, lp);

            return BitmapUtility.FromByteArray(result, bounds.ViewportWidth, bounds.ViewportHeight);
        }

        // GPU: Allocating Memory on GPU only!
        internal static Image RenderGpu2(Bounds bounds)
        {
            var deviceResult = Gpu.Default.Allocate<byte>(3 * bounds.ViewportWidth * bounds.ViewportHeight);
            bounds.AdjustAspectRatio();
            var lp = ComputeLaunchParameters(bounds);
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Gpu.Default.Launch(() =>
            {
                var x = blockDim.x * blockIdx.x + threadIdx.x;
                var y = blockDim.y * blockIdx.y + threadIdx.y;
                var offset = 3 * (y * bounds.ViewportWidth + x);

                var c = new Complex
                {
                    Real      = bounds.XMin + x * scale,
                    Imaginary = bounds.YMin + y * scale,
                };

                ComputeJuliaSetAtOffset(deviceResult, c, offset);
            }, lp);

            return BitmapUtility.FromByteArray(Gpu.CopyToHost(deviceResult), bounds.ViewportWidth, bounds.ViewportHeight);
        }

        // GPU: Parallel.For!
        internal static Image RenderGpu3(Bounds bounds)
        {
            var result = new byte[3 * bounds.ViewportWidth * bounds.ViewportHeight];
            bounds.AdjustAspectRatio();
            var scale = (bounds.XMax - bounds.XMin) / bounds.ViewportWidth;

            Gpu.Default.For(0, bounds.ViewportWidth * bounds.ViewportHeight, i =>
            {
                var x = i % bounds.ViewportWidth;
                var y = i / bounds.ViewportWidth;
                var offset = 3 * i;

                var c = new Complex
                {
                    Real      = bounds.XMin + x * scale,
                    Imaginary = bounds.YMin + y * scale,
                };

                ComputeJuliaSetAtOffset(result, c, offset);
            });

            return BitmapUtility.FromByteArray(result, bounds.ViewportWidth, bounds.ViewportHeight);
        }

        // ReSharper disable once SuggestBaseTypeForParameter
        private static void ComputeJuliaSetAtOffset(byte[] result, Complex a, int offset)
        {
            var c = new Complex
            {
                Real      = -0.8f,
                Imaginary = +0.156f,
            };

            for (byte i = 0; i < 255; ++i)
            {
                a = a * a + c;

                if (a.Magnitude() >= 2)
                {
                    result[offset + 0] = i;
                    result[offset + 1] = i;
                    result[offset + 2] = i;
                    break;
                }
            }
        }

        private static LaunchParam ComputeLaunchParameters(Bounds bounds)
        {
            const int tpb = 32;
            var width = bounds.ViewportWidth;
            var height = bounds.ViewportHeight;
            return new LaunchParam(new dim3((width + (tpb - 1)) / tpb, (height + (tpb - 1)) / tpb), new dim3(tpb, tpb));
        }
    }
}