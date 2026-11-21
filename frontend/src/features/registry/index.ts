import { Library } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const registryFeature: FeatureDef = {
  id: "registry",
  nav: { label: "Registry", icon: Library, path: "/registry", section: "Models", order: 60 },
  routes: [{ path: "registry", ...lazyPage(() => import("./RegistryPage")) }],
};
