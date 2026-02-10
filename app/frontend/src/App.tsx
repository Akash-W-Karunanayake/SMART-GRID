import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import Simulation from './pages/Simulation'
import SelfHealing from './pages/SelfHealing'
import Forecasting from './pages/Forecasting'
import Diagnostics from './pages/Diagnostics'
import NetLoad from './pages/NetLoad'
import Settings from './pages/Settings'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="simulation" element={<Simulation />} />
          <Route path="self-healing" element={<SelfHealing />} />
          <Route path="forecasting" element={<Forecasting />} />
          <Route path="diagnostics" element={<Diagnostics />} />
          <Route path="net-load" element={<NetLoad />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
