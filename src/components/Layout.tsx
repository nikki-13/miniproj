
import { Navbar } from "@/components/Navbar";
import { Outlet } from "react-router-dom";

export function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t py-6 md:py-8">
        <div className="container flex flex-col items-center justify-center gap-4 text-center md:flex-row">
          <p className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} X-ray Insight. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
