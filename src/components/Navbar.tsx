
import { useState } from "react";
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Activity, Menu, X } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";

const navItems = [
  { path: "/", label: "Home" },
  { path: "/about", label: "About" },
  { path: "/upload", label: "Upload X-ray" },
  { path: "/team", label: "Our Team" },
];

export function Navbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const isMobile = useIsMobile();

  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-2">
          <NavLink 
            to="/" 
            className="flex items-center gap-2 font-semibold text-lg text-foreground"
          >
            <Activity className="h-6 w-6 text-medical-600" />
            <span>X-ray Insight</span>
          </NavLink>
        </div>

        {!isMobile && (
          <nav className="flex items-center gap-6">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  cn(
                    "text-sm font-medium transition-colors hover:text-primary",
                    isActive ? "text-foreground" : "text-muted-foreground"
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
            <ThemeToggle />
          </nav>
        )}

        {isMobile && (
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={toggleMobileMenu}
            >
              {isMobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
              <span className="sr-only">Toggle menu</span>
            </Button>
          </div>
        )}
      </div>

      {isMobile && isMobileMenuOpen && (
        <nav className="container pb-4 md:hidden">
          <ul className="flex flex-col space-y-3">
            {navItems.map((item) => (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  className={({ isActive }) =>
                    cn(
                      "block py-2 text-sm font-medium transition-colors hover:text-primary",
                      isActive ? "text-foreground" : "text-muted-foreground"
                    )
                  }
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </header>
  );
}
