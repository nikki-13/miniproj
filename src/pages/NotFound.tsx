
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { AlertCircle } from "lucide-react";

export default function NotFound() {
  return (
    <div className="container flex flex-col items-center justify-center min-h-[80vh] px-4 text-center">
      <AlertCircle className="h-16 w-16 text-medical-600 mb-6" />
      <h1 className="text-4xl font-bold tracking-tighter mb-4">404 - Page Not Found</h1>
      <p className="text-xl text-muted-foreground max-w-md mb-8">
        Sorry, the page you are looking for does not exist or has been moved.
      </p>
      <Link to="/">
        <Button size="lg">
          Return to Home
        </Button>
      </Link>
    </div>
  );
}
