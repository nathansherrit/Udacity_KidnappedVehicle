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
#include <chrono> 

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{	
	default_random_engine gen;
	gen.seed(chrono::system_clock::now().time_since_epoch().count());
	num_particles = 300;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 0;
		particles.push_back(p);
	}
	
	is_initialized = true;	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	default_random_engine gen;
	gen.seed(chrono::system_clock::now().time_since_epoch().count());
  
  	for (int i = 0; i < num_particles; i++)
    {
      	if(yaw_rate == 0) // Prevents division by 0 error
        	yaw_rate = 0.000001;
      
		double newX = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
		double newY = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
        double newTheta = particles[i].theta + yaw_rate*delta_t;

        normal_distribution<double> dist_x(newX, std_pos[0]);
        normal_distribution<double> dist_y(newY, std_pos[1]);
        normal_distribution<double> dist_theta(newTheta, std_pos[2]);
      	
      	particles[i].x = dist_x(gen);
      	particles[i].y = dist_y(gen);
      	particles[i].theta = dist_theta(gen);      
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (unsigned int i = 0; i < predicted.size(); i++)
	{
		if (observations.size() >= 1)
		{
			double minDist = dist(predicted[i].x, predicted[i].y, observations[0].x, observations[0].y);
			double minIndex = 0;
			for (int j = 1; j < observations.size(); j++)
			{
				double distance = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);
				if (distance < minDist)
				{
					minDist = distance;
					minIndex = j;
				}
			}

			// After closest neighbor is found, pair the observation with the prediction and remove observation from list?
			predicted[i].id = observations[minIndex].id;
			observations.erase(observations.begin() + minIndex);
		}
	}



}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	

	//Transform observations to map cooridantaes
	double totalWeight = 0;
	for (int i = 0; i < num_particles; i++)
	{
		vector<LandmarkObs> observations_transformed;
		particles[i].weight = 1;

      	// Convert each ovservations to map coordinates
		for (int j = 0; j < static_cast<unsigned int>(observations.size()); j++)
		{
          
			LandmarkObs newObs;
			double newX = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
			double newY = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
			newObs.x = newX;
			newObs.y = newY;
			newObs.id = 0;
			observations_transformed.push_back(newObs);
		}

		// Find the landmark that is closest to the observation
		for (int k = 0; k < static_cast<unsigned int>(observations_transformed.size()); k++)
		{			
			map_landmarks.landmark_list.size();

			double minDist = dist(map_landmarks.landmark_list[0].x_f, map_landmarks.landmark_list[0].y_f, observations_transformed[k].x, observations_transformed[k].y);
			double minIndex = 0;
			for (unsigned int j = 1; j < map_landmarks.landmark_list.size(); j++)
			{
				double distance = dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, observations_transformed[k].x, observations_transformed[k].y);
				if (distance < minDist)
				{                  
					minDist = distance;
					minIndex = j;
				}
			}

			// After closest neighbor is found, pair the observation with the landmark
          	if(minDist <= sensor_range)
            {
				observations_transformed[k].id = map_landmarks.landmark_list[minIndex].id_i;
                double x1, y1, x2, y2;
                x1 = observations_transformed[k].x; 
          		y1 = observations_transformed[k].y; 
          		x2 = map_landmarks.landmark_list[minIndex].x_f;
          		y2 = map_landmarks.landmark_list[minIndex].y_f;

          		double a = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
          		double b = -1 * (pow(x1 - x2, 2) / (2 * pow(std_landmark[0], 2)) + pow(y1 - y2, 2) / (2 * pow(std_landmark[1], 2)));
          		double ans = a * exp(b);
          		particles[i].weight *= ans; 
            }
          
          	else
            {
            	observations_transformed[k].id = -1;
          		particles[i].weight *= 0; 		
            }
		}
      	totalWeight += particles[i].weight;
	}
    
  	// Normalize all the weights	
	for (unsigned int i = 0; i < num_particles; i++)
    {
		particles[i].weight /= totalWeight;	
    }
}

void ParticleFilter::resample() 
{
	vector<Particle> newParticles;
	double r = ((double)rand() / (RAND_MAX));
	int index = floor(r * num_particles);
	double beta = 0;
	double mw = 0;
	for (int i = 0; i < num_particles; i++)
	{
		if (particles[i].weight > mw)
			mw = particles[i].weight;
	}

	for (int i = 0; i < num_particles; i++)
	{
		r = ((double)rand() / (RAND_MAX));
		beta += r * 2.0 * mw;
		while (beta > particles[index].weight)
		{
			beta -= particles[index].weight;
			index = (index + 1) % num_particles;
		}
		newParticles.push_back(particles[index]);
	}

	particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
	const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

void ParticleFilter::printParticles()
{
  	for(int i = 0; i < static_cast<unsigned int>(particles.size()); i++)
    	{
      		cout << i << ": " << particles[i].x << ", " << particles[i].y << 
              ", " << particles[i].theta << ", " << particles[i].weight <<  endl;
    	}
}