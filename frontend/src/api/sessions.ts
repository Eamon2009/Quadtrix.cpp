import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "./client";
import type { Message, Session } from "../types";

export function useSessions() {
  return useQuery({
    queryKey: ["sessions"],
    queryFn: async (): Promise<Session[]> => {
      try {
        return await apiFetch<Session[]>("/api/sessions");
      } catch (error) {
        throw error;
      }
    },
  });
}

export function useCreateSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (title?: string): Promise<Session> => {
      try {
        return await apiFetch<Session>("/api/sessions", {
          method: "POST",
          body: JSON.stringify({ title }),
        });
      } catch (error) {
        throw error;
      }
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}

export function useDeleteSession() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (sessionId: string): Promise<void> => {
      try {
        await apiFetch<void>(`/api/sessions/${sessionId}`, { method: "DELETE" });
      } catch (error) {
        throw error;
      }
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}

export function useSessionMessages(sessionId: string | null) {
  return useQuery({
    queryKey: ["messages", sessionId],
    enabled: Boolean(sessionId),
    queryFn: async (): Promise<Message[]> => {
      try {
        return await apiFetch<Message[]>(`/api/sessions/${sessionId}/messages`);
      } catch (error) {
        throw error;
      }
    },
  });
}
