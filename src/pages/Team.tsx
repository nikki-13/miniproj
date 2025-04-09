
import { Card } from "@/components/ui/card";
import { User } from "lucide-react";

export default function Team() {
  const teamMembers = [
    {
      id: 1,
      name: "Team Member 1",
      role: "Role / Position",
      bio: "Short biography or description about the team member and their expertise in the project.",
      image: null, // Will be filled later
    },
    {
      id: 2,
      name: "Team Member 2",
      role: "Role / Position",
      bio: "Short biography or description about the team member and their expertise in the project.",
      image: null, // Will be filled later
    },
    {
      id: 3,
      name: "Team Member 3",
      role: "Role / Position",
      bio: "Short biography or description about the team member and their expertise in the project.",
      image: null, // Will be filled later
    },
  ];

  return (
    <div className="container max-w-5xl px-4 py-12 md:py-16 lg:py-24">
      <div className="flex flex-col items-center justify-center space-y-6 text-center">
        <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
          Our Team
        </h1>
        <p className="max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400">
          Meet the team behind X-ray Insight - the individuals who developed this pneumonia detection tool.
        </p>
      </div>

      <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-8">
        {teamMembers.map((member) => (
          <Card key={member.id} className="overflow-hidden border shadow-sm">
            <div className="flex flex-col items-center p-6 text-center">
              <div className="relative mb-4 h-24 w-24 rounded-full bg-gray-100 flex items-center justify-center">
                {member.image ? (
                  <img
                    src={member.image}
                    alt={member.name}
                    className="h-full w-full rounded-full object-cover"
                  />
                ) : (
                  <User className="h-12 w-12 text-gray-400" />
                )}
              </div>
              <h3 className="text-xl font-bold">{member.name}</h3>
              <p className="text-sm text-medical-600 font-medium mt-1">{member.role}</p>
              <p className="mt-4 text-sm text-muted-foreground">{member.bio}</p>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
