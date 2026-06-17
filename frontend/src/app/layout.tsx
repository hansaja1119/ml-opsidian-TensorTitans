import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';

export const metadata: Metadata = {
  title: 'FloodSim — Urban Planning Simulation Tool',
  description: 'Simulate flood risk scenarios across Sri Lankan districts. Powered by champion ML ensemble models from the ML Opsidian Genesis competition.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="app-container">
          <Navbar />
          {children}
        </div>
      </body>
    </html>
  );
}
