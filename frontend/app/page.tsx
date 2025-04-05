"use client"; // Required for components with client-side interactions or hooks (like iframe)

import { Button } from "@/components/ui/button"; // Import the shadcn Button

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8 font-[family-name:var(--font-geist-sans)]">
      <header className="mb-8 text-center w-full max-w-5xl"> {/* Increased max-width */}
        <h1 className="text-4xl font-bold tracking-tight">DataVision Food & Theft Detection</h1> {/* Updated Title */}
        <p className="text-lg text-muted-foreground mt-2">A minimalistic demo integrating YOLO food detection and theft/scanning analysis.</p> {/* Updated Description */}
      </header>

      {/* Adjusted main container for better layout */}
      <main className="flex flex-col gap-6 items-center w-full max-w-5xl border rounded-lg p-6 shadow-sm bg-card"> {/* Increased max-width and added bg */}
        <h2 className="text-2xl font-semibold">Interactive Demo</h2> {/* Simplified heading */}
        <p className="text-sm text-center text-muted-foreground">
          The Gradio interface below provides tools for food detection and self-checkout video analysis. It's served from the Python backend running on port 7860.
        </p> {/* Updated description */}

        {/* Embed the Gradio app using an iframe */}
        <iframe
          src="http://localhost:7860" // URL where the Gradio app runs
          width="100%"
          height="850px" // Increased height significantly
          className="border rounded-md shadow-inner" // Added inner shadow
          title="DataVision Demo Interface" // Updated title
        ></iframe>
        {/* Removed the extra Next.js button section */}
      </main>

      <footer className="mt-8 text-center text-sm text-muted-foreground">
        A-star25 Hackathon Project | Powered by Next.js, Gradio, YOLO.
      </footer> {/* Updated footer */}
    </div>
  );
}
