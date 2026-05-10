import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Satellite, Shield, BarChart3 } from 'lucide-react';
import { motion } from 'framer-motion';

const LandingPage = () => {
  const navigate = useNavigate();
  const [mousePos, setMousePos] = useState({ x: -1000, y: -1000 });

  const handleMouseMove = (e) => {
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseLeave = () => {
    setMousePos({ x: -1000, y: -1000 });
  };

  return (
    <div
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="landing-container min-h-screen relative flex flex-col items-center justify-center p-8 overflow-hidden"
      style={{ backgroundColor: '#fbf9f4' }}
    >
      {/* Base Grid Layer */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: 'linear-gradient(to right, #CBD5E0 1px, transparent 1px), linear-gradient(to bottom, #CBD5E0 1px, transparent 1px)',
        backgroundSize: '40px 40px',
        opacity: 0.3,
      }} />

      {/* Interactive Bulge Layer */}
      <div style={{
        position: 'absolute', inset: -40,
        backgroundImage: 'linear-gradient(to right, #b60058 1px, transparent 1px), linear-gradient(to bottom, #b60058 1px, transparent 1px)',
        backgroundSize: '40px 40px',
        transform: 'scale(1.03)',
        WebkitMaskImage: `radial-gradient(circle 350px at ${mousePos.x}px ${mousePos.y}px, black 0%, transparent 100%)`,
        maskImage: `radial-gradient(circle 350px at ${mousePos.x}px ${mousePos.y}px, black 0%, transparent 100%)`,
        pointerEvents: 'none',
        opacity: 0.4,
      }} />
      {/* Decorative Crosshair */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full pointer-events-none opacity-20">
        <div className="absolute top-1/2 left-0 w-full h-[1px] bg-slate-400"></div>
        <div className="absolute top-0 left-1/2 w-[1px] h-full bg-slate-400"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] border border-slate-400 rounded-full"></div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-10 max-w-4xl w-full text-center"
      >
        <div className="mono text-[11px] uppercase tracking-[0.2em] text-primary mb-4 flex items-center justify-center gap-2">
          <Satellite size={14} />
          Computer vision System // Ver 1.0.4
        </div>

        <h1 className="text-[64px] leading-tight mb-6 text-slate">
          Arctic Precision <br />
          <span className="text-primary-bright">Vision</span>
        </h1>

        <p className="text-lg text-on-surface-variant max-w-2xl mx-auto mb-12">
          High-precision semantic segmentation for glacial lakes and cryospheric anomalies.
          Powered by multi-sensor fusion and state-of-the-art deep learning architectures.
        </p>

        <div className="flex items-center justify-center gap-6">
          <button
            onClick={() => navigate('/dashboard')}
            className="btn-primary flex items-center gap-2 group"
          >
            Launch Dashboard
            <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
          </button>

          <button
            onClick={() => navigate('/comparison')}
            className="btn-outline"
          >
            View Model Comparison
          </button>
        </div>
      </motion.div>

      {/* Feature Strip */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5, duration: 1 }}
        className="absolute bottom-12 w-full max-w-6xl px-8 flex justify-between gap-8"
      >
        <div className="flex-1 p-6 sharp-border bg-white/50 backdrop-blur-sm">
          <Shield className="text-primary-bright mb-3" size={24} />
          <h3 className="mono text-sm uppercase tracking-wider mb-2">Scientific Rigor</h3>
          <p className="text-sm text-on-surface-variant">Validated against Himalayan field data with &gt;95% F1 precision.</p>
        </div>

        <div className="flex-1 p-6 sharp-border bg-white/50 backdrop-blur-sm">
          <BarChart3 className="text-primary-bright mb-3" size={24} />
          <h3 className="mono text-sm uppercase tracking-wider mb-2">Multi-Model Labs</h3>
          <p className="text-sm text-on-surface-variant">Compare U-Net, Simple CNN, and ASPP-SegNet architectures.</p>
        </div>

        <div className="flex-1 p-6 sharp-border bg-white/50 backdrop-blur-sm">
          <Satellite className="text-primary-bright mb-3" size={24} />
          <h3 className="mono text-sm uppercase tracking-wider mb-2">Sentinel-2 Ready</h3>
          <p className="text-sm text-on-surface-variant">Native support for multi-spectral band patches and TIF formats.</p>
        </div>
      </motion.div>

      {/* Coordinate Telemetry */}
      <div className="absolute top-8 right-8 mono text-[10px] text-on-surface-variant space-y-1 text-right">
        <div>LAT: 28.5983° N</div>
        <div>LON: 83.9310° E</div>
        <div>ALT: 5,420M AMSL</div>
      </div>

      <div className="absolute top-8 left-8 mono text-[10px] text-on-surface-variant">
        <div>Mohammad Ibrahim : 23I0083</div>
        <div>Mohammad Hamza : 23I0047</div>
      </div>
    </div>
  );
};

export default LandingPage;
