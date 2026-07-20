import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Whisper Docker",
  description: "GPU-first speech transcription control room",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

