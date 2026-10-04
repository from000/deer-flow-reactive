import { afterEach, describe, expect, it } from "@rstest/core";
import { act, cleanup, screen } from "@testing-library/react";
import { hydrateRoot, type Root } from "react-dom/client";
import { renderToString } from "react-dom/server";

import {
  Sidebar,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";

type HappyDOMWindow = typeof window & {
  happyDOM: { setViewport: (viewport: { width?: number }) => void };
};

function setViewportWidth(width: number) {
  (window as HappyDOMWindow).happyDOM.setViewport({ width });
}

function SidebarHarness() {
  return (
    <SidebarProvider>
      <Sidebar>
        <button type="button">New chat</button>
      </Sidebar>
      <SidebarTrigger />
    </SidebarProvider>
  );
}

afterEach(() => {
  cleanup();
  setViewportWidth(1024);
});

describe("SidebarProvider mobile first click", () => {
  it("keeps the SSR trigger disabled until React can receive its first click", async () => {
    setViewportWidth(375);
    const container = document.createElement("div");
    container.innerHTML = renderToString(<SidebarHarness />);
    document.body.append(container);

    const serverTrigger = container.querySelector<HTMLButtonElement>(
      '[data-sidebar="trigger"]',
    );
    expect(serverTrigger?.disabled).toBe(true);

    let root: Root | undefined;
    await act(async () => {
      root = hydrateRoot(container, <SidebarHarness />);
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    });

    const trigger = screen.getByRole("button", { name: /toggle sidebar/i });
    expect((trigger as HTMLButtonElement).disabled).toBe(false);
    root?.unmount();
  });
});
