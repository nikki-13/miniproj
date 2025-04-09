
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowRight, Stethoscope, BarChart, Clock } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="w-full py-12 md:py-24 lg:py-32 bg-gradient-to-b from-background to-medical-100 dark:from-background dark:to-medical-900/20">
        <div className="container px-4 md:px-6 space-y-10 xl:space-y-16">
          <div className="grid gap-8 lg:grid-cols-2 lg:gap-12 xl:gap-16">
            <div className="space-y-6 animate-fade-in">
              <div className="inline-block rounded-lg bg-medical-100 dark:bg-medical-900/20 px-3 py-1 text-sm text-medical-800 dark:text-medical-300">
                Introducing X-ray Insight
              </div>
              <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
                AI-Powered Pneumonia Detection
              </h1>
              <p className="max-w-[600px] text-gray-500 md:text-xl dark:text-gray-400">
                Upload chest X-rays and get instant analysis with our
                state-of-the-art deep learning model. Fast, accurate,
                and easy to use.
              </p>
              <div className="flex flex-col gap-3 sm:flex-row">
                <Link to="/upload">
                  <Button size="lg" className="group">
                    Upload X-ray
                    <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                  </Button>
                </Link>
                <Link to="/about">
                  <Button variant="outline" size="lg">
                    Learn More
                  </Button>
                </Link>
              </div>
            </div>
            <div className="flex items-center justify-center">
              <div className="relative w-full max-w-[500px] aspect-square overflow-hidden rounded-xl shadow-2xl animate-fade-in animate-delay-200">
                <div className="absolute inset-0 bg-gradient-to-tr from-medical-600/20 via-medical-500/10 to-medical-400/5 backdrop-blur-sm"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-4/5 h-4/5 border-2 border-medical-500/50 rounded-lg flex items-center justify-center">
                    <div className="text-medical-500 text-opacity-80 text-sm text-center px-4">
                      Upload your X-ray image to detect pneumonia
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="w-full py-12 md:py-24 bg-background">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <div className="inline-block rounded-lg bg-medical-100 dark:bg-medical-900/20 px-3 py-1 text-sm text-medical-800 dark:text-medical-300">
                Key Features
              </div>
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                Why Use X-ray Insight?
              </h2>
              <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                Our platform combines cutting-edge AI with an intuitive interface to deliver accurate pneumonia detection
              </p>
            </div>
          </div>
          <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-3 lg:gap-12 mt-12">
            {/* Feature 1 */}
            <div className="flex flex-col items-center space-y-4 animate-fade-in">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-medical-100 dark:bg-medical-900/20">
                <Stethoscope className="h-8 w-8 text-medical-600" />
              </div>
              <h3 className="text-xl font-bold">Accurate Results</h3>
              <p className="text-center text-gray-500 dark:text-gray-400">
                Our model achieves over 95% accuracy in detecting pneumonia from chest X-rays
              </p>
            </div>
            {/* Feature 2 */}
            <div className="flex flex-col items-center space-y-4 animate-fade-in animate-delay-200">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-medical-100 dark:bg-medical-900/20">
                <Clock className="h-8 w-8 text-medical-600" />
              </div>
              <h3 className="text-xl font-bold">Instant Analysis</h3>
              <p className="text-center text-gray-500 dark:text-gray-400">
                Get results within seconds of uploading your X-ray image
              </p>
            </div>
            {/* Feature 3 */}
            <div className="flex flex-col items-center space-y-4 animate-fade-in animate-delay-300">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-medical-100 dark:bg-medical-900/20">
                <BarChart className="h-8 w-8 text-medical-600" />
              </div>
              <h3 className="text-xl font-bold">Advanced Model</h3>
              <p className="text-center text-gray-500 dark:text-gray-400">
                Powered by Swin Transformer architecture, the latest in medical image analysis
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="w-full py-12 md:py-24 bg-medical-50 dark:bg-medical-900/10">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                Ready to Try X-ray Insight?
              </h2>
              <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                Upload a chest X-ray and get pneumonia detection results in seconds
              </p>
            </div>
            <div className="mx-auto w-full max-w-sm space-y-2">
              <Link to="/upload" className="w-full">
                <Button size="lg" className="w-full">
                  Upload X-ray Now
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
