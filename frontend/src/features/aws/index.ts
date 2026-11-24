import { Cloud } from "lucide-react";
import { lazyPage, type FeatureDef } from "@/app/feature";

export const awsFeature: FeatureDef = {
  id: "aws",
  nav: { label: "AWS deployment", icon: Cloud, path: "/aws", section: "Serve", order: 85 },
  routes: [{ path: "aws", ...lazyPage(() => import("./AwsPage")) }],
};
