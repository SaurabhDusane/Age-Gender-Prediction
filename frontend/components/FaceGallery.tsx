'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { Search, Filter } from 'lucide-react';

export default function FaceGallery({ apiUrl }: { apiUrl: string }) {
  const [faces, setFaces] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({
    age_min: '',
    age_max: '',
    gender: '',
    emotion: '',
  });

  const fetchFaces = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filters.age_min) params.append('age_min', filters.age_min);
      if (filters.age_max) params.append('age_max', filters.age_max);
      if (filters.gender) params.append('gender', filters.gender);
      if (filters.emotion) params.append('emotion', filters.emotion);
      params.append('limit', '50');

      const response = await axios.get(
        `${apiUrl}/api/v1/faces/search?${params.toString()}`
      );
      setFaces(response.data);
    } catch (error) {
      console.error('Failed to fetch faces:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFaces();
  }, []);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Face Gallery</h2>
        <button
          onClick={fetchFaces}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          <Search className="h-5 w-5 mr-2" />
          Search
        </button>
      </div>

      <div className="mb-6 grid grid-cols-1 md:grid-cols-4 gap-4">
        <input
          type="number"
          placeholder="Min Age"
          value={filters.age_min}
          onChange={(e) => setFilters({ ...filters, age_min: e.target.value })}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <input
          type="number"
          placeholder="Max Age"
          value={filters.age_max}
          onChange={(e) => setFilters({ ...filters, age_max: e.target.value })}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <select
          value={filters.gender}
          onChange={(e) => setFilters({ ...filters, gender: e.target.value })}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="">All Genders</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>
        <select
          value={filters.emotion}
          onChange={(e) => setFilters({ ...filters, emotion: e.target.value })}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="">All Emotions</option>
          <option value="happy">Happy</option>
          <option value="sad">Sad</option>
          <option value="angry">Angry</option>
          <option value="neutral">Neutral</option>
          <option value="surprise">Surprise</option>
        </select>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin h-12 w-12 border-4 border-blue-600 border-t-transparent rounded-full mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading faces...</p>
        </div>
      ) : faces.length === 0 ? (
        <div className="text-center py-12 text-gray-500">
          <Filter className="h-12 w-12 mx-auto mb-4" />
          <p>No faces found matching your criteria</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {faces.map((face) => (
            <div
              key={face.id}
              className="border border-gray-200 rounded-lg p-4 hover:shadow-lg transition"
            >
              <div className="space-y-2 text-sm">
                <div className="font-semibold text-gray-900">
                  {new Date(face.timestamp).toLocaleDateString()}
                </div>
                <div className="text-gray-600">
                  <div>Age: {face.age?.toFixed(1)} years</div>
                  <div>Gender: {face.gender}</div>
                  <div>Emotion: {face.emotion}</div>
                  {face.ethnicity && <div>Ethnicity: {face.ethnicity}</div>}
                  <div className="text-xs text-gray-400 mt-2">
                    Quality: {(face.quality_score * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
