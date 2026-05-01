import { useSettingsStore } from "../store/settingsStore";
import type { ApiError } from "../types";

export class ApiClientError extends Error {
  apiError: ApiError;

  constructor(apiError: ApiError) {
    super(apiError.message);
    this.apiError = apiError;
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const baseUrl = useSettingsStore.getState().apiBaseUrl.replace(/\/$/, "");
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      ...init,
      headers,
    });
    if (!response.ok) {
      const fallback: ApiError = { error: "request_failed", message: response.statusText, code: response.status };
      const error = (await response.json().catch(() => fallback)) as ApiError;
      throw new ApiClientError(error);
    }
    if (response.status === 204) {
      return undefined as T;
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiClientError) {
      throw error;
    }
    throw new ApiClientError({ error: "network_error", message: "Could not reach the API server", code: 0 });
  }
}
