import { Link, useLocation } from 'react-router-dom';
import { signOut } from 'firebase/auth';
import { auth } from '../firebase';
import { useAuth } from '../hooks/useAuth';

export default function Navbar() {
  const { user } = useAuth();
  const loc = useLocation();

  const navLink = (to, label) => (
    <Link to={to} style={{
      fontSize: 13, fontWeight: 500, padding: '6px 14px', borderRadius: 6,
      color: loc.pathname === to ? 'var(--accent2)' : 'var(--text2)',
      background: loc.pathname === to ? 'rgba(124,106,247,0.12)' : 'transparent',
      transition: 'all 0.15s'
    }}>{label}</Link>
  );

  return (
    <nav style={{
      position: 'sticky', top: 0, zIndex: 100,
      borderBottom: '1px solid var(--border)',
      background: 'rgba(10,10,15,0.85)',
      backdropFilter: 'blur(12px)',
      display: 'flex', alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px', height: 56
    }}>
      <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700,
          color: 'var(--accent2)', letterSpacing: '-0.02em'
        }}>
          {'<Muhammad />'}
        </span>
        <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
          LC Journal
        </span>
      </Link>

      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        {navLink('/', 'Journal')}
        {user && navLink('/admin', 'Dashboard')}
        {user
          ? <button
              onClick={() => signOut(auth)}
              style={{ fontSize: 13, color: 'var(--text3)', padding: '6px 14px', marginLeft: 8 }}
            >Sign out</button>
          : <Link to="/login" className="btn btn-primary" style={{ fontSize: 13, padding: '6px 16px', marginLeft: 8 }}>
              Admin
            </Link>
        }
      </div>
    </nav>
  );
}
