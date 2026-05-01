import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";
import type { HealthResponse, ModelStats } from "../types";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: async (): Promise<HealthResponse> => {
      try {
        return await apiFetch<HealthResponse>("/api/health");
      } catch (error) {
        throw error;
      }
    },
    refetchInterval: 30000,
    retry: 1,
  });
}

export function useStats() {
  return useQuery({
    queryKey: ["stats"],
    queryFn: async (): Promise<ModelStats> => {
      try {
        return await apiFetch<ModelStats>("/api/stats");
      } catch (error) {
        throw error;
      }
    },
    refetchInterval: 30000,
    retry: 1,
  });
}
