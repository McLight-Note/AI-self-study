import { useEffect, useState } from 'react';
import { collection, getDocs } from 'firebase/firestore';
import { db } from '../firebase';
import { useAuth } from '../hooks/useAuth';
import { useNavigate } from 'react-router-dom';
import { PROBLEMS, MONTH_NAMES } from '../data/problems';
import SolutionEditor from '../components/SolutionEditor';

const STATUS_COLOR = { solved: 'var(--easy)', attempted: 'var(--medium)', 'needs-review': 'var(--red)' };

export default function Admin() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [solutions, setSolutions] = useState({});
  const [fetching, setFetching] = useState(true);
  const [selectedProblem, setSelectedProblem] = useState(null);
  const [filterMonth, setFilterMonth] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [search, setSearch] = useState('');

  useEffect(() => {
    if (!loading && !user) navigate('/login');
  }, [user, loading]);

  const loadSolutions = async () => {
    try {
      const snap = await getDocs(collection(db, 'solutions'));
      const map = {};
      snap.forEach(d => { map[d.id] = d.data(); });
      setSolutions(map);
    } catch (e) { console.error(e); }
    finally { setFetching(false); }
  };

  useEffect(() => { if (user) loadSolutions(); }, [user]);

  const solved = Object.values(solutions).filter(s => s.status === 'solved').length;

  const filtered = PROBLEMS.filter(p => {
    if (filterMonth !== 'all' && p.month !== Number(filterMonth)) return false;
    if (filterStatus === 'solved' && solutions[p.id]?.status !== 'solved') return false;
    if (filterStatus === 'unsolved' && solutions[p.id]) return false;
    if (filterStatus === 'attempted' && solutions[p.id]?.status !== 'attempted') return false;
    if (search && !p.title.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  if (loading || fetching) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '80vh', color: 'var(--text3)' }}>
      Loading...
    </div>
  );

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 20px', display: 'flex', gap: 24 }}>

      {/* Left: problem list */}
      <div style={{ width: 380, flexShrink: 0 }}>
        <div style={{ marginBottom: 20 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 4 }}>Admin Dashboard</h2>
          <p style={{ fontSize: 13, color: 'var(--text3)' }}>
            {solved}/150 solved · {Math.round(solved / 150 * 100)}% complete
          </p>
          <div style={{ height: 4, background: 'var(--bg3)', borderRadius: 2, marginTop: 10, overflow: 'hidden' }}>
            <div style={{
              height: '100%', width: `${solved / 150 * 100}%`,
              background: 'linear-gradient(90deg, var(--accent), var(--accent2))',
              borderRadius: 2, transition: 'width 0.5s'
            }} />
          </div>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search problems..."
            style={{ fontSize: 13 }}
          />
          <div style={{ display: 'flex', gap: 8 }}>
            <select value={filterMonth} onChange={e => setFilterMonth(e.target.value)} style={{ flex: 1, fontSize: 13 }}>
              <option value="all">All months</option>
              {[1,2,3,4,5,6].map(m => <option key={m} value={m}>Month {m}</option>)}
            </select>
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} style={{ flex: 1, fontSize: 13 }}>
              <option value="all">All status</option>
              <option value="solved">Solved</option>
              <option value="attempted">Attempted</option>
              <option value="unsolved">Unsolved</option>
            </select>
          </div>
        </div>

        {/* Problem list */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 'calc(100vh - 280px)', overflowY: 'auto' }}>
          {filtered.map(p => {
            const sol = solutions[p.id];
            const isSelected = selectedProblem?.id === p.id;
            return (
              <div
                key={p.id}
                onClick={() => setSelectedProblem(p)}
                style={{
                  padding: '10px 14px', borderRadius: 8, cursor: 'pointer',
                  background: isSelected ? 'rgba(124,106,247,0.15)' : 'var(--surface)',
                  border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
                  transition: 'all 0.15s',
                  display: 'flex', alignItems: 'center', gap: 10
                }}
              >
                <div style={{
                  width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                  background: sol ? STATUS_COLOR[sol.status] : 'var(--border2)'
                }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {p.title}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text3)' }}>M{p.month} · {p.topic}</div>
                </div>
                <span style={{
                  fontSize: 10, fontWeight: 700, fontFamily: 'var(--font-mono)',
                  color: p.difficulty === 'Easy' ? 'var(--easy)' : p.difficulty === 'Medium' ? 'var(--medium)' : 'var(--red)'
                }}>
                  {p.difficulty[0]}
                </span>
              </div>
            );
          })}
          {filtered.length === 0 && (
            <div style={{ color: 'var(--text3)', textAlign: 'center', padding: 32, fontSize: 14 }}>
              No problems match filters
            </div>
          )}
        </div>
      </div>

      {/* Right: editor */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {selectedProblem ? (
          <div className="card fade-up" style={{ padding: 28 }}>
            {/* Problem header */}
            <div style={{ marginBottom: 24, paddingBottom: 20, borderBottom: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <span className={`badge badge-${selectedProblem.difficulty.toLowerCase()}`}>
                  {selectedProblem.difficulty}
                </span>
                <span style={{ fontSize: 12, color: 'var(--text3)' }}>
                  Month {selectedProblem.month} · {selectedProblem.topic}
                </span>
                <a
                  href={selectedProblem.leetcodeUrl}
                  target="_blank"
                  rel="noreferrer"
                  style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--accent2)' }}
                >
                  Open on LeetCode ↗
                </a>
              </div>
              <h2 style={{ fontSize: 20, fontWeight: 600 }}>{selectedProblem.title}</h2>
            </div>

            <SolutionEditor
              problem={selectedProblem}
              existing={solutions[selectedProblem.id]}
              onSaved={loadSolutions}
            />
          </div>
        ) : (
          <div style={{
            height: '100%', display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            color: 'var(--text3)', gap: 12
          }}>
            <div style={{ fontSize: 48 }}>{'</>'}</div>
            <p style={{ fontSize: 14 }}>Select a problem to add your solution</p>
          </div>
        )}
      </div>
    </div>
  );
}
