import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:12345',
});

export const runSvcPredict = payload => api.post(`/svc-predict-another`, payload);
export const runKnnPredict = payload => api.post(`/knn-predict-another`, payload)

const apis = {
  runSvcPredict,
  runKnnPredict
};

export default apis