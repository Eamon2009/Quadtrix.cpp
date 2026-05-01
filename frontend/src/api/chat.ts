import { useMutation, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "./client";
import type { ChatRequest, ChatResponse } from "../types";

export function useSendMessage() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (payload: ChatRequest): Promise<ChatResponse> => {
      try {
        return await apiFetch<ChatResponse>("/api/chat", {
          method: "POST",
          body: JSON.stringify(payload),
        });
      } catch (error) {
        throw error;
      }
    },
    onSuccess: async (response) => {
      await queryClient.invalidateQueries({ queryKey: ["sessions"] });
      await queryClient.invalidateQueries({ queryKey: ["messages", response.session_id] });
    },
  });
}
