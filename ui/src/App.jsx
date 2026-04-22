import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/layout/Sidebar'
import Header from './components/layout/Header'
import Dashboard from './pages/Dashboard'
import EpisodeArena from './pages/EpisodeArena'
import TaskBank from './pages/TaskBank'
import ForgeQueue from './pages/ForgeQueue'
import OversightStats from './pages/OversightStats'
import DifficultyCurve from './pages/DifficultyCurve'
import StandaloneGrader from './pages/StandaloneGrader'

export default function App() {
  return (
<div className="flex h-screen overflow-hidden bg-base">
      {/* Background overlay removed */}

      <Sidebar />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/arena" element={<EpisodeArena />} />
            <Route path="/tasks" element={<TaskBank />} />
            <Route path="/forge" element={<ForgeQueue />} />
            <Route path="/oversight" element={<OversightStats />} />
            <Route path="/difficulty" element={<DifficultyCurve />} />
            <Route path="/grader" element={<StandaloneGrader />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
