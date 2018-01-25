/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// For each particle make prediction vector as subset of landmarks vector
void makePredictionVector(const Particle& particle, const double& sensor_range, const Map& map_landmarks,
	vector<LandmarkObs>& prediction, vector<int>& landmark_id)
{
	double range = sensor_range * sensor_range;

	prediction.clear();
	landmark_id.clear();

	int idx = 0;
	auto landmark_num = map_landmarks.landmark_list.size();
	for (int j = 0; j < landmark_num; ++j)
	{
		const Map::single_landmark_s& landmark = map_landmarks.landmark_list[j];
		double dx = landmark.x_f - particle.x;
		double dy = landmark.y_f - particle.y;

		if ((dx * dx + dy * dy) <= range)
		{
			++idx;

			LandmarkObs pred;
			pred.id = idx;
			pred.x = landmark.x_f;
			pred.y = landmark.y_f;

			prediction.push_back(pred);
			landmark_id.push_back(landmark.id_i);
		}
	}
}

// Transform observation into the world coordinates
void getObservationWorldPosition(Particle& particle, const LandmarkObs& observation, LandmarkObs& obs_world)
{
	double costheta = cos(particle.theta);
	double sintheta = sin(particle.theta);

	double world_x = particle.x + costheta * observation.x - sintheta * observation.y;
	double world_y = particle.y + sintheta * observation.x + costheta * observation.y;

	obs_world.id = 0;
	obs_world.x = world_x;
	obs_world.y = world_y;
}

// Multivariate Gaussian
double gauss2d(const double& x, const double& mux, const double& stdx, const double& y, const double& muy, const double& stdy)
{
	try
	{
		double dx = (x - mux) / stdx;
		double dy = (y - muy) / stdy;

		return 0.5 / M_PI / stdx / stdy * exp(-0.5 * (dx * dx + dy * dy));
	}
	catch (exception& ex)
	{
	}

	return 0;
}

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// Number of partical enougth for accuracy and performance compromise.
	num_particles = 100;

	// Generate normally distributed particles accross initial location and position

	default_random_engine generator;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		Particle particle;
		particle.id = i + 1;
		particle.x = dist_x(generator);
		particle.y = dist_y(generator);
		particle.theta = dist_theta(generator);
		particle.weight = 1.0;

		particles.push_back(particle);
	}

	// And mark filter as initialized
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// Some usefult variables
	double& dt = delta_t;
	double dt2 = delta_t * delta_t;

	// Random generator with 0 mean, will be added to new values
	default_random_engine generator;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		Particle& particle = particles[i];

		double& theta = particle.theta;

		double new_x;
		double new_y;
		double new_theta;

		new_theta = theta + yaw_rate * dt;

		if (fabs(yaw_rate) < 0.0001)
		{
			new_x = particle.x + velocity * cos(theta) * dt;
			new_y = particle.y + velocity * sin(theta) * dt;
		}
		else
		{
			new_x = particle.x + velocity / yaw_rate * (sin(new_theta) - sin(theta));
			new_y = particle.y + velocity / yaw_rate * (cos(theta) - cos(new_theta));
		}

		new_x += dist_x(generator);
		new_y += dist_y(generator);
		new_theta += dist_theta(generator);

		particle.x = new_x;
		particle.y = new_y;
		particle.theta = new_theta;
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs>& predicted, vector<LandmarkObs>& observations)
{
	auto obs_num = observations.size();
	auto pred_num = predicted.size();
	for (int i = 0; i < obs_num; ++i)
	{
		LandmarkObs& observation = observations[i];
		
		int best_id = 0;
		double best_dist = 0;

		for (int j = 0; j < pred_num; ++j)
		{
			LandmarkObs& pred = predicted[j];

			double dx = pred.x - observation.x;
			double dy = pred.y - observation.y;
			double dist = dx * dx + dy * dy;

			if (best_id <= 0 || dist < best_dist)
			{
				best_id = pred.id;
				best_dist = dist;
			}
		}

		if (best_id > 0)
		{
			observation.id = best_id;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	auto obs_num = observations.size();

	// Skip this step if no any observations
	if (obs_num <= 0)
	{
		return;
	}

	vector<LandmarkObs> prediction;
	vector<int> landmark_id;
	bool prediction_initialized = false;
	double sensor_range_upd = sensor_range * 1.2;
	double fromx = 0;
	double fromy = 0;
	double tox = 0;
	double toy = 0;

	double weight_sum = 0;
	for (int i = 0; i < num_particles; ++i)
	{
		Particle& particle = particles[i];
		vector<LandmarkObs> obs_world;

		// Transform observations into world coordinates
		for (int j = 0; j < obs_num; ++j)
		{
			LandmarkObs obs_w;
			getObservationWorldPosition(particle, observations[j], obs_w);
			obs_world.push_back(obs_w);
		}

		// Select possible landmarks in working range of sensor and associate it with observations
		if (!prediction_initialized ||
			((particle.x - sensor_range) < fromx) ||
			((particle.y - sensor_range) < fromy) ||
			((particle.x + sensor_range) > tox) ||
			((particle.y + sensor_range) > toy))
		{
			makePredictionVector(particle, sensor_range_upd, map_landmarks, prediction, landmark_id);
			prediction_initialized = true;
			fromx = particle.x - sensor_range_upd;
			fromy = particle.y - sensor_range_upd;
			tox = particle.x + sensor_range_upd;
			toy = particle.y + sensor_range_upd;
		}

		dataAssociation(prediction, obs_world);

		// Update weight for each particle
		double weight = 1.0;
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

		for (int j = 0; j < obs_num; ++j)
		{
			LandmarkObs& observation = obs_world[j];
			if (observation.id > 0)
			{
				LandmarkObs& pred = prediction[observation.id - 1];

				associations.push_back(landmark_id[observation.id - 1]);
				sense_x.push_back(observation.x);
				sense_y.push_back(observation.y);

				weight *= gauss2d(pred.x, observation.x, std_landmark[0], pred.y, observation.y, std_landmark[1]);
			}
		}

		weight_sum += weight;

		particle.weight = weight;

		// Set output data for simulator
		SetAssociations(particle, associations, sense_x, sense_y);
	}

	// Normalize weights
	for (int i = 0; i < num_particles; ++i)
	{
		Particle& particle = particles[i];

		if (weight_sum > 0)
		{
			particle.weight /= weight_sum;
		}
		else
		{
			particle.weight = 1.0;
		}
	}
}

void ParticleFilter::resample()
{
	// Combine weight vector for discrete normal distribution
	vector<double> weight_vec;
	for (int i = 0; i < num_particles; ++i)
	{
		Particle& particle = particles[i];
		weight_vec.push_back(particle.weight);
	}

	discrete_distribution<> rnd(weight_vec.begin(), weight_vec.end());
	default_random_engine generator;

	vector<Particle> new_particles;
	for (int i = 0; i < num_particles; ++i)
	{
		int idx = rnd(generator);

		new_particles.push_back(particles[idx]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const vector<int>& associations, 
                                     const vector<double>& sense_x, const vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
