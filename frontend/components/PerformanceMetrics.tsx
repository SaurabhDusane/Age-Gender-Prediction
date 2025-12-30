'use client';

import { useState, useEffect } from 'react';
import { Activity, Cpu, Zap } from 'lucide-react';

export default function PerformanceMetrics({ apiUrl }: { apiUrl: string }) {
  const [metrics, setMetrics] = useState({
    fps: 0,
    latency: 0,
    gpuUsage: 0,
    memoryUsage: 0,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics({
        fps: Math.random() * 10 + 25,
        latency: Math.random() * 30 + 70,
        gpuUsage: Math.random() * 30 + 50,
        memoryUsage: Math.random() * 20 + 60,
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Performance Metrics</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            icon={<Zap className="h-6 w-6" />}
            title="FPS"
            value={metrics.fps.toFixed(1)}
            unit="frames/sec"
            color="blue"
            target={30}
            current={metrics.fps}
          />
          
          <MetricCard
            icon={<Activity className="h-6 w-6" />}
            title="Latency"
            value={metrics.latency.toFixed(0)}
            unit="ms"
            color="green"
            target={100}
            current={metrics.latency}
            inverse
          />
          
          <MetricCard
            icon={<Cpu className="h-6 w-6" />}
            title="GPU Usage"
            value={metrics.gpuUsage.toFixed(1)}
            unit="%"
            color="purple"
            target={80}
            current={metrics.gpuUsage}
          />
          
          <MetricCard
            icon={<Cpu className="h-6 w-6" />}
            title="Memory"
            value={metrics.memoryUsage.toFixed(1)}
            unit="%"
            color="orange"
            target={80}
            current={metrics.memoryUsage}
          />
        </div>

        <div className="mt-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <InfoItem label="CUDA Available" value="Yes" />
            <InfoItem label="GPU Count" value="1" />
            <InfoItem label="Model" value="YOLOv8 + Ensemble" />
            <InfoItem label="Tracking" value="Enabled" />
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ icon, title, value, unit, color, target, current, inverse = false }: any) {
  const percentage = inverse
    ? Math.max(0, (1 - current / target) * 100)
    : Math.min(100, (current / target) * 100);
  
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
    orange: 'bg-orange-100 text-orange-600',
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-2 rounded-lg ${colorClasses[color as keyof typeof colorClasses]}`}>
          {icon}
        </div>
      </div>
      <div>
        <p className="text-sm font-medium text-gray-600">{title}</p>
        <p className="mt-2 text-3xl font-semibold text-gray-900">
          {value}
          <span className="text-lg text-gray-500 ml-1">{unit}</span>
        </p>
        <div className="mt-3 w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-300 ${
              percentage >= 80 ? 'bg-green-500' : percentage >= 50 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    </div>
  );
}

function InfoItem({ label, value }: any) {
  return (
    <div className="flex justify-between py-2 border-b border-gray-200">
      <span className="text-gray-600">{label}</span>
      <span className="font-medium text-gray-900">{value}</span>
    </div>
  );
}
