'use client';

import { Bar, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

export default function Statistics({ stats }: { stats: any }) {
  if (!stats) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <p className="text-gray-500">Loading statistics...</p>
      </div>
    );
  }

  const genderData = {
    labels: Object.keys(stats.gender_distribution || {}),
    datasets: [
      {
        label: 'Gender Distribution',
        data: Object.values(stats.gender_distribution || {}),
        backgroundColor: ['rgba(54, 162, 235, 0.8)', 'rgba(255, 99, 132, 0.8)'],
        borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)'],
        borderWidth: 1,
      },
    ],
  };

  const emotionData = {
    labels: Object.keys(stats.emotion_distribution || {}),
    datasets: [
      {
        label: 'Emotion Distribution',
        data: Object.values(stats.emotion_distribution || {}),
        backgroundColor: 'rgba(75, 192, 192, 0.8)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Statistics Overview</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-blue-50 rounded-lg p-4">
            <p className="text-sm font-medium text-blue-900">Total Faces</p>
            <p className="mt-2 text-3xl font-semibold text-blue-600">
              {stats.total_faces || 0}
            </p>
          </div>
          
          <div className="bg-green-50 rounded-lg p-4">
            <p className="text-sm font-medium text-green-900">Average Age</p>
            <p className="mt-2 text-3xl font-semibold text-green-600">
              {stats.avg_age?.toFixed(1) || 0}
            </p>
          </div>
          
          <div className="bg-purple-50 rounded-lg p-4">
            <p className="text-sm font-medium text-purple-900">Unique Sessions</p>
            <p className="mt-2 text-3xl font-semibold text-purple-600">--</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Gender Distribution</h3>
            <div className="h-64">
              <Pie data={genderData} options={{ maintainAspectRatio: false }} />
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Emotion Distribution</h3>
            <div className="h-64">
              <Bar
                data={emotionData}
                options={{
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                    },
                  },
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
