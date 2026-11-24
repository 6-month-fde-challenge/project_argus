import { FileSliders } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const profilesFeature: FeatureDef = {
  id: "profiles",
  nav: { label: "Profiles", icon: FileSliders, path: "/profiles", section: "Models", order: 45 },
  routes: [{ path: "profiles", ...lazyPage(() => import("./ProfilesPage")) }],
};
