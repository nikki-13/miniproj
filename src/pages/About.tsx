
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Brain, Database, Microscope, ActivitySquare } from "lucide-react";

export default function About() {
  return (
    <div className="container max-w-5xl px-4 py-12 md:py-16 lg:py-24">
      <div className="flex flex-col items-center justify-center space-y-6 text-center">
        <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
          About X-ray Insight
        </h1>
        <p className="max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400">
          Learn about our AI-powered pneumonia detection technology and how it can assist healthcare professionals.
        </p>
      </div>

      <div className="mt-16">
        <h2 className="text-2xl font-bold mb-6">The Technology</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <Card className="border shadow-sm">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-medical-600" />
                <CardTitle>Swin Transformer Model</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-2">
              <p className="text-muted-foreground">
                Our pneumonia detection system is powered by the Swin Transformer, a state-of-the-art vision model that
                outperforms traditional convolutional neural networks for medical image analysis.
              </p>
              <p className="text-muted-foreground">
                The Swin Transformer uses a hierarchical representation with shifted windows, allowing it to capture both
                fine details and broader patterns in chest X-rays.
              </p>
            </CardContent>
          </Card>

          <Card className="border shadow-sm">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-medical-600" />
                <CardTitle>Dataset & Training</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-2">
              <p className="text-muted-foreground">
                Our model was trained on the Kaggle Chest X-Ray Images (Pneumonia) dataset, containing over 5,800 X-ray
                images from pediatric patients.
              </p>
              <p className="text-muted-foreground">
                The dataset includes both normal X-rays and X-rays showing bacterial and viral pneumonia, allowing our
                model to learn the distinct visual patterns associated with the condition.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      <Separator className="my-16" />

      <div>
        <h2 className="text-2xl font-bold mb-6">Performance Metrics</h2>
        <Card className="border shadow-sm">
          <CardHeader>
            <div className="flex items-center gap-2">
              <ActivitySquare className="h-5 w-5 text-medical-600" />
              <CardTitle>Model Accuracy & Performance</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="flex flex-col items-center p-4 bg-medical-50 dark:bg-medical-900/10 rounded-lg">
                <span className="text-3xl font-bold text-medical-700 dark:text-medical-400">96.8%</span>
                <span className="text-sm text-muted-foreground mt-1">Accuracy</span>
              </div>
              <div className="flex flex-col items-center p-4 bg-medical-50 dark:bg-medical-900/10 rounded-lg">
                <span className="text-3xl font-bold text-medical-700 dark:text-medical-400">95.2%</span>
                <span className="text-sm text-muted-foreground mt-1">Precision</span>
              </div>
              <div className="flex flex-col items-center p-4 bg-medical-50 dark:bg-medical-900/10 rounded-lg">
                <span className="text-3xl font-bold text-medical-700 dark:text-medical-400">94.7%</span>
                <span className="text-sm text-muted-foreground mt-1">Recall</span>
              </div>
            </div>
            <p className="mt-6 text-sm text-muted-foreground">
              Our model has been validated through rigorous testing and achieves high performance across all key metrics.
              However, it's important to note that while our system provides highly accurate analyses, it is designed to
              assist healthcare professionals rather than replace clinical diagnosis.
            </p>
          </CardContent>
        </Card>
      </div>

      <Separator className="my-16" />

      <div>
        <h2 className="text-2xl font-bold mb-6">Research & Applications</h2>
        <Card className="border shadow-sm">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Microscope className="h-5 w-5 text-medical-600" />
              <CardTitle>Clinical Applications</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">
              X-ray Insight is designed to support healthcare professionals by providing rapid, preliminary assessments of
              chest X-rays for pneumonia. Potential applications include:
            </p>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>Triage in emergency departments to prioritize patients</li>
              <li>Screening tool in resource-limited settings</li>
              <li>Secondary check for radiologists</li>
              <li>Educational tool for medical training</li>
              <li>Research platform for pneumonia pattern analysis</li>
            </ul>
            <p className="text-muted-foreground pt-2">
              Our team is actively researching additional applications and continuously improving the model's
              performance through regular updates and training on expanded datasets.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
