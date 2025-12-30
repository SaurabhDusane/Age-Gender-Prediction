'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { Camera, Users, Activity, TrendingUp } from 'lucide-react';
import LiveStream from '../components/LiveStream';
import Statistics from '../components/Statistics';
import FaceGallery from '../components/FaceGallery';
import PerformanceMetrics from '../components/PerformanceMetrics';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [activeTab, setActiveTab] = useState('live');
  const [stats, setStats] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    checkConnection();
    fetchStatistics();
    const interval = setInterval(fetchStatistics, 5000);
    return () => clearInterval(interval);
  }, []);

  const checkConnection = async () => {
    try {
      await axios.get(`${API_URL}/api/v1/health`);
      setIsConnected(true);
    } catch (error) {
      setIsConnected(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/faces/statistics`);
      setStats(response.data);
    } catch (error) {
      console.error('Failed to fetch statistics:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center">
              <Users className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">
                Demographic Analysis System
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="ml-2 text-sm text-gray-600">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <StatCard
            icon={<Users className="h-6 w-6" />}
            title="Total Faces"
            value={stats?.total_faces || 0}
            color="blue"
          />
          <StatCard
            icon={<TrendingUp className="h-6 w-6" />}
            title="Avg Age"
            value={stats?.avg_age?.toFixed(1) || 0}
            color="green"
          />
          <StatCard
            icon={<Activity className="h-6 w-6" />}
            title="Sessions"
            value="--"
            color="purple"
          />
          <StatCard
            icon={<Camera className="h-6 w-6" />}
            title="Active Streams"
            value="--"
            color="orange"
          />
        </div>

        <div className="mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              <TabButton
                active={activeTab === 'live'}
                onClick={() => setActiveTab('live')}
                label="Live Stream"
              />
              <TabButton
                active={activeTab === 'statistics'}
                onClick={() => setActiveTab('statistics')}
                label="Statistics"
              />
              <TabButton
                active={activeTab === 'gallery'}
                onClick={() => setActiveTab('gallery')}
                label="Face Gallery"
              />
              <TabButton
                active={activeTab === 'performance'}
                onClick={() => setActiveTab('performance')}
                label="Performance"
              />
            </nav>
          </div>
        </div>

        <div>
          {activeTab === 'live' && <LiveStream apiUrl={API_URL} />}
          {activeTab === 'statistics' && <Statistics stats={stats} />}
          {activeTab === 'gallery' && <FaceGallery apiUrl={API_URL} />}
          {activeTab === 'performance' && <PerformanceMetrics apiUrl={API_URL} />}
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, title, value, color }: any) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
    orange: 'bg-orange-100 text-orange-600',
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="mt-2 text-3xl font-semibold text-gray-900">{value}</p>
        </div>
        <div className={`p-3 rounded-full ${colorClasses[color as keyof typeof colorClasses]}`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

function TabButton({ active, onClick, label }: any) {
  return (
    <button
      onClick={onClick}
      className={`${
        active
          ? 'border-blue-500 text-blue-600'
          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
      } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
    >
      {label}
    </button>
  );
}
