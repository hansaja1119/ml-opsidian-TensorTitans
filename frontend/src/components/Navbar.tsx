'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Waves, Zap, BarChart3, Lightbulb, GitCompareArrows, MapPinned } from 'lucide-react';

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <Link href="/" className="navbar-brand">
          <div className="logo-icon">
            <Waves size={18} color="#fff" />
          </div>
          <span>FloodSim</span>
        </Link>

        <ul className="navbar-links">
          <li>
            <Link href="/" className={pathname === '/' ? 'active' : ''}>
              <Zap size={16} /> Simulator
            </Link>
          </li>
          <li>
            <Link href="/interventions" className={pathname === '/interventions' ? 'active' : ''}>
              <Lightbulb size={16} /> Interventions
            </Link>
          </li>
          <li>
            <Link href="/compare" className={pathname === '/compare' ? 'active' : ''}>
              <GitCompareArrows size={16} /> Compare
            </Link>
          </li>
          <li>
            <Link href="/districts" className={pathname === '/districts' ? 'active' : ''}>
              <MapPinned size={16} /> Districts
            </Link>
          </li>
          <li>
            <Link href="/analytics" className={pathname === '/analytics' ? 'active' : ''}>
              <BarChart3 size={16} /> Analytics
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}
