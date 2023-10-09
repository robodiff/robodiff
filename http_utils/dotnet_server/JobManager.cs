//
//  Copyright 2021  David Matthews
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.DependencyInjection;
using System.IO;
using System.IO.Compression;

using K4os.Compression.LZ4.Streams;

namespace MinimalAPI
{
    public static class MagicNumbers
    {
        readonly public static int ErrorCountUntilFailure = 10;
        readonly public static TimeSpan TimeoutTimeSpan = TimeSpan.FromMinutes(10);

        readonly public static int DefaultRetryDelaySeconds = 10;
    }

    public record ExportFolderName(string folderName, bool compression);
    public record JobSubmissionRecord(dynamic[] InnerSimulations, dynamic? JobDetails);

//    public record JobResults(Guid JobGuid, dynamic?[] SimulationResults);

    public record SimulationRecord(Guid JobGuid, int SimIdx, int RetryCount);
    public record DispatchedWorkRecord(Guid WorkGuid, Guid NodeGuid, SimulationRecord SimRecord);
    public record SimulationDetails(Guid WorkGuid, dynamic? InnerSimulation, int RetryDelay);
    public class NodeWorkInfo 
    {
        public Guid WorkGuid {get; set;}
        public DateTime StartTime {get; set;}
        public DateTime FinishTime {get; set;}
        public int CheckInCount {get; set;} = 0;
        public ResultsStatuses Status {get; set;} = ResultsStatuses.PENDING;

        public NodeWorkInfo(Guid workGuid) {
            WorkGuid = workGuid;
            StartTime = DateTime.Now;
        }
        public void UpdateWork() {
            CheckInCount++;
        }
        public void FinishWork() {
            FinishTime = DateTime.Now;
            Status = ResultsStatuses.COMPLETE;
        }
        public void ErrorWork() {
            Status = ResultsStatuses.FAILED;
        }

    }
    public class NodeInfo
    {
        private int _errorCnt;
        public Guid NodeGuid { get; private set; }
        public int ErrorCount { get => _errorCnt; }
        public dynamic? Details { get; private set; }
        public DateTime LastContactTime { get; private set; }
        public bool NodeIsShutdown { get; private set; } = false;
        public List<NodeWorkInfo> WorkLog {get; private set;} = new();
        private NodeWorkInfo? currentWork;
        public NodeInfo(Guid nodeGuid, int errorCount, dynamic? details)
        {
            NodeGuid = nodeGuid;
            _errorCnt = errorCount;
            Details = details;
            LastContactTime = DateTime.Now;
        }

        public void StartNewWork(Guid workGuid) {
            currentWork = new(workGuid);
            WorkLog.Add(currentWork);
        }
        public void UpdateWork() {
            currentWork?.UpdateWork();
            UpdateLastContactTime();
        }
        public int ErrorWork() {
            currentWork?.ErrorWork();
            UpdateLastContactTime();
            return IncrementErrors();
        }
        public void FinishWork() {
            currentWork?.FinishWork();
            UpdateLastContactTime();
        }

        public void ShutdownNode()
        {
            UpdateLastContactTime();
            NodeIsShutdown = true;
        }

        public bool IsNodeActive()
        {
            return !NodeIsShutdown & LastContactTime + MagicNumbers.TimeoutTimeSpan >= DateTime.Now;
        }


        /// <summary>
        /// threadsafe increment of the total error count.
        /// </summary>
        /// <returns>new total error cnt for this node</returns>
        private int IncrementErrors()
        {
            return Interlocked.Increment(ref _errorCnt);
        }
        private void UpdateLastContactTime()
        {
            LastContactTime = DateTime.Now;
        }

    }

    public record ExportSimulationResultsRecord(Guid WorkGuid, 
                                                DateTime LastContact,
                                                ResultsStatuses Status,
                                                dynamic[] Results,
                                                dynamic? Errors);

    public record ExportSimulationRecord(dynamic InnerSim, ExportSimulationResultsRecord? Results);

    public record ExportSimulationRecordDetailed(dynamic InnerSim, ExportSimulationResultsRecord[] Results);

    
    public enum ResultsStatuses
    {
        PENDING,
        COMPLETE,
        FAILED
    }

    public class SimulationResults
    {

        private List<dynamic> resultsList;
        private ResultsStatuses resultsStatus;
        private DateTime lastContact = new();
        private dynamic? errorDetails = default;
        private object _lock = new();

        public ResultsStatuses ResultsStatus
        {
            get
            {
                lock(_lock)
                {
                    return resultsStatus;
                }
            }
            set
            {
                lock(_lock)
                {
                    resultsStatus = value;
                    lastContact = DateTime.Now;
                }
            }
        } // get set

        public string ResultsStatusStr
        {
            get => ResultsStatus.ToString();
        } // get

        public DateTime LastContact
        {
            get
            {
                lock(_lock)
                {
                    return lastContact;
                }
            }
        } // get

        public dynamic? ErrorDetails
        {
            get
            {
                lock(_lock)
                {
                    return errorDetails;
                }
            }
            set
            {
                lock(_lock)
                {
                    errorDetails = value;
                }
            }
        } // get set


        public bool CheckAndMarkTimeout(TimeSpan timeoutTimeSpan) {
            lock(_lock) {
                if (resultsStatus == ResultsStatuses.PENDING && lastContact + timeoutTimeSpan < DateTime.Now) {
                    resultsStatus = ResultsStatuses.FAILED;
                    return true;
                }
                return false;
            }
        }

        public void AppendResults(dynamic partialResults)
        {
            lock (_lock)
            {
                resultsList.Add(partialResults);
                lastContact = DateTime.Now;
            }
        }

        public SimulationResults()
        {
            lock(_lock)
            {
                resultsList = new();
            }
            ResultsStatus = ResultsStatuses.PENDING;
        }
        private dynamic[] GetResults()
        {
            lock(_lock)
            {
                return resultsList.ToArray();
            }
        }

        public ExportSimulationResultsRecord Export(Guid workGuid)
        {
            return new ExportSimulationResultsRecord(workGuid, LastContact, ResultsStatus, GetResults(), ErrorDetails);
        }
    }

    public class Simulation
    {
        private dynamic _innerSim;
        private Dictionary<Guid, SimulationResults> simResults = new();
        private object _lock = new();

        public dynamic InnerSim
        {
            get
            {
                lock(_lock)
                {
                    return _innerSim;
                }
            }
        } // get

        private KeyValuePair<Guid, SimulationResults>[] FetchSimulationResults()
        {
            lock(_lock)
            {
                return simResults.ToArray();
            }

        }

        public Simulation(dynamic innerSim)
        {
            lock(_lock)
            {
                _innerSim = innerSim;
            }
        }


        public dynamic? InitSimResults(Guid workGuid)
        {
            lock(_lock)
            {
                simResults[workGuid] = new SimulationResults();
                return _innerSim;
            }
        }

        public SimulationResults? GetSimulationResults(Guid workGuid)
        {
            try
            {
                lock(_lock)
                {
                    return simResults[workGuid];
                }
            }
            catch
            {
                return default;
            }
        }

        public bool IsCompleted()
        { 
            return FetchSimulationResults().Any(kv => kv.Value.ResultsStatus == ResultsStatuses.COMPLETE);
        }


        public ExportSimulationRecord? GetCompletedResults()
        {

            foreach (var kv in FetchSimulationResults())
            {
                if (kv.Value.ResultsStatus == ResultsStatuses.COMPLETE)
                {
                    return new ExportSimulationRecord(InnerSim, kv.Value.Export(kv.Key));
                }
            }
            return default;
        }

        public ExportSimulationRecordDetailed GetAllResults()
        {
            ExportSimulationResultsRecord[] resultsRecords = FetchSimulationResults()
                                                               .Select(kv => kv.Value.Export(kv.Key))
                                                               .ToArray();
            return new ExportSimulationRecordDetailed(InnerSim, resultsRecords);
        }
    }


    public record ExportJobRecord(Guid JobGuid,
                                  dynamic? JobDetails,
                                  int CurrSimIdx,
                                  int CompletedSimulationCount,
                                  int TotalSimulationCount);

    public class Job
    {
        private Guid jobGuid = new();
        private dynamic? _jobDetails = default;
        private readonly Simulation[] simulations; // does not require locking. Object is readonly after construction!
        private int currSimIdx = 0;
        private object _lock = new();

        public Guid JobGuid
        {
            get
            {
                lock(_lock)
                {
                    return jobGuid;
                }
            }
        } // get
        public int CurrSimIdx
        {
            get
            {
                lock(_lock)
                {
                    return currSimIdx;
                }
            }
        } // get 
        public dynamic? JobDetails
        {
            get
            {
                lock(_lock)
                {
                    return _jobDetails;
                }
            }
        } // get

        public Job (dynamic [] innerSimulations, dynamic? jobDetails = default )
        {
            var sims = innerSimulations.Select(x => new Simulation(x)).ToArray();
            lock(_lock)
            {
                simulations = sims;
                jobGuid = Guid.NewGuid();
                _jobDetails = jobDetails;
            }
        }

        public Simulation? GetSimulation(int simIdx)
        {
            if (simIdx < simulations.Length)
            {
                return simulations[simIdx];
            }
            return default;
        }

        public dynamic? GetInnerSim(int simIdx)
        {
            return GetSimulation(simIdx)?.InnerSim ?? default;
        }

        public SimulationResults? GetSimulationResults(int simIdx, Guid workGuid)
        {
            return GetSimulation(simIdx)?.GetSimulationResults(workGuid) ?? default;
        }

        public bool GetNextSimulationRecord(out SimulationRecord? simRecord)
        {
            int? nextSimIdx = default;
            lock(_lock)
            {
                if (currSimIdx < simulations.Length)
                {
                    nextSimIdx = currSimIdx++;
                }
            }
            if (nextSimIdx.HasValue)
            {
                simRecord  = new SimulationRecord(JobGuid, nextSimIdx!.Value, 0);
            }
            else
            {
                simRecord = default;
            }

            return simRecord != default;
        }

        public ExportJobRecord ExportJobSummary()
        {
            int totalSimCount = simulations.Length;
            int completedSimCount = simulations.Where(sim => sim.IsCompleted()).Count();

            return new ExportJobRecord(JobGuid, JobDetails, CurrSimIdx, completedSimCount, totalSimCount);
        }

        public void ExportCompletedSimulationsFolder(string folder, bool compression = true)
        {
            Directory.CreateDirectory($"{folder}/{JobGuid}");
            if (compression) {
                using (var fileOut = new GZipStream(File.Create($"{folder}/{JobGuid}/completedSimulations.json.gz"), CompressionMode.Compress))
                ExportCompletedSimulationsFile(fileOut);
            }
            else {
                using (var fileOut = File.Create($"{folder}/{JobGuid}/completedSimulations.json"))
                ExportCompletedSimulationsFile(fileOut);
            }
        }

        public void ExportAllSimulationsFolder(string folder, bool compression = true)
        {
            Directory.CreateDirectory($"{folder}/{JobGuid}");
            if (compression) {
                using (var fileOut = new GZipStream(File.Create($"{folder}/{JobGuid}/allSimulations.json.gz"), CompressionMode.Compress))
                ExportAllSimulationsFile(fileOut);
            }
            else {
                using (var fileOut = File.Create($"{folder}/{JobGuid}/allSimulations.json"))
                ExportAllSimulationsFile(fileOut);
            }
        }

        public void ExportCompletedSimulationsFile(Stream fileOut)
        {
            byte[] newline = new UTF8Encoding().GetBytes("\n");

            for (int simExportIdx = 0; simExportIdx < simulations.Length; simExportIdx++)
            {
                ExportSimulationRecord? record  = simulations[simExportIdx].GetCompletedResults();
                if (record == default) continue;
                byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(record);
                fileOut.Write(bytes, 0, bytes.Length);
                fileOut.Write(newline, 0, newline.Length);
            }
        }

        public void ExportAllSimulationsFile(Stream fileOut)
        {
            byte[] newline = new UTF8Encoding().GetBytes("\n");

            for (int simExportIdx = 0; simExportIdx < simulations.Length; simExportIdx++)
            {
                var record  = simulations[simExportIdx].GetAllResults();
                if (record == default) continue;
                byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(record);
                fileOut.Write(bytes, 0, bytes.Length);
                fileOut.Write(newline, 0, newline.Length);
            }
        }
    }

    public record StatsRecord(int TotalJobCount = 0,
                                int DispatchedJobCount = 0,
                                int TotalSimulationCount = 0,
                                int DispatchedSimulationCount = 0,
                                int OutstandingSimulationCount = 0,
                                int TotalSimulationErrorCount = 0,
                                int FailedSimulationCount = 0,
                                int TotalResultsReceivedCount = 0,
                                int TotalFinishedSimulationCount = 0,
                                int TotalNodeCount = 0,
                                int ActiveNodeCount = 0,
                                int SimulationsInRetryQueue = 0);

    public class JobRepository
    {
        readonly object jobsDictLock = new();
        readonly Dictionary<Guid, Job> jobsDict = new();

        readonly object primaryQueueLock = new();
        readonly List<Job>  primaryQueue = new();

        public volatile int CurrentJobIdx = 0;
        public volatile int TotalSimulationCount = 0;

        public volatile int DispatchedSimulationCount = 0;
        public volatile int TotalSimulationErrorCount = 0;
        public volatile int FailedSimulationCount = 0;
        public volatile int TotalResultsReceivedCount = 0;
        public volatile int TotalFinishedSimulationCount = 0;

        readonly object outstandingWorkDictLock = new();
        readonly Dictionary<Guid, DispatchedWorkRecord> outstandingWorkDict = new();

        readonly ConcurrentQueue<SimulationRecord> retryQueue = new();

        readonly object nodeInfoDictLock = new();
        readonly Dictionary<Guid, NodeInfo> nodeInfoDict = new();

        public Job GetJobFromJobsDict(Guid jobGuid)
        {
            Job job;
            lock(jobsDictLock)
            {
                job = jobsDict[jobGuid];
            }
            return job;
        }

        /// <summary>
        /// Threadsafe method to remove workRecords from the outstanding dictionary.
        /// </summary>
        /// <param name="workGuid"></param>
        /// <param name="workRecord">Place to put workRecord if it exists.</param>
        /// <returns>true if removal was successful, false otherwise </returns>
        public bool RemoveOutstandingWork(Guid workGuid, out DispatchedWorkRecord? workRecord)
        {
            bool removeSuccess;
            lock(outstandingWorkDictLock)
            {
                removeSuccess = outstandingWorkDict.Remove(workGuid, out workRecord);
            }

            return removeSuccess;
        }

        #region REGISTER_NODE
        public void ShutdownNode(Guid nodeGuid)
        {
            NodeInfo nodeInfo;
            lock(nodeInfoDictLock)
            {
                nodeInfo = nodeInfoDict[nodeGuid];
            }
            nodeInfo.ShutdownNode();
        }

        public Guid RegisterNode(dynamic? nodeDetails = default)
        {
            Guid nodeGuid = Guid.NewGuid();
            lock(nodeInfoDictLock)
            {
                nodeInfoDict[nodeGuid] = new NodeInfo(nodeGuid, 0, details: nodeDetails);
            }
            return nodeGuid;
        }
        #endregion // REGISTER_NODE

        #region SUBMIT_JOB
        public int GetPrimaryQueueCount()
        {
            int primaryQueueCount;
            lock(primaryQueueLock)
            {
                primaryQueueCount = primaryQueue.Count;
            }
            return primaryQueueCount;
        }

        public Guid EnqueueJob(dynamic [] innerSimulations, dynamic? jobDetails = default)
        {
            Job job = new(innerSimulations, jobDetails: jobDetails);
            lock (jobsDictLock)
            {
                jobsDict[job.JobGuid] = job;
            }
            lock(primaryQueueLock)
            {
                primaryQueue.Add(job);
                TotalSimulationCount += innerSimulations.Length;
            }
            return job.JobGuid;
        }
        #endregion // SUBMIT_JOB

        #region GET_A_SIMULATION
        public bool TryGetPendingSimulation(out SimulationRecord? simRecord)
        {
            simRecord = default;
            while (CurrentJobIdx < GetPrimaryQueueCount())
            {
                Job currentJob;
                lock(primaryQueueLock)
                {
                    currentJob = primaryQueue[CurrentJobIdx];
                }

                if (currentJob.GetNextSimulationRecord(out simRecord))
                {
                    return true;
                }

                else
                {
                    Interlocked.Increment(ref CurrentJobIdx);
                }
            }

            return false;
        }

        public bool TryGetRetrySimulation(out SimulationRecord? simRecord)
        {
            return retryQueue.TryDequeue(out simRecord);
        }

        public bool IdentifyTimedOutOutstandingWork()
        {
            bool existsSimsToRetry = false;
            List<KeyValuePair<Guid, DispatchedWorkRecord>> outstandingWorkList;
            lock(outstandingWorkDictLock)
            {
                outstandingWorkList = outstandingWorkDict.ToList();
            }

            foreach (var kv in outstandingWorkList)
            {
                SimulationRecord simRecord = kv.Value.SimRecord;
                SimulationResults? simResults = GetJobFromJobsDict(simRecord.JobGuid).GetSimulationResults(simRecord.SimIdx, kv.Key);

                DateTime lastResultsContact = simResults!.LastContact;

                if (simResults.CheckAndMarkTimeout(MagicNumbers.TimeoutTimeSpan)) // this is atomic! needs to be!
                {
                    if (simRecord.RetryCount < MagicNumbers.ErrorCountUntilFailure)
                    {
                        existsSimsToRetry = true;
                        retryQueue.Enqueue(new SimulationRecord(simRecord.JobGuid, simRecord.SimIdx, simRecord.RetryCount + 1));
                    }
                    else
                    {
                        Interlocked.Increment(ref FailedSimulationCount);
                    }
                    Interlocked.Increment(ref TotalSimulationErrorCount);
                }
            }
            return existsSimsToRetry;
        }

        public SimulationDetails DispatchSimulation(SimulationRecord simulationRecord, Guid nodeGuid)
        {

            // Record that work is being dispatched.
            // Also initialize the results section of the simulation for the current workGuid.
            // + Init last contact time

            Guid workGuid = Guid.NewGuid();
            NodeInfo? nodeInfo;
            lock(nodeInfoDictLock)
            {
                nodeInfoDict.TryGetValue(nodeGuid, out nodeInfo);
            }
            nodeInfo!.StartNewWork(workGuid);

            DispatchedWorkRecord dispatchedWorkRecord = new(workGuid, nodeGuid, simulationRecord);

            lock(outstandingWorkDictLock)
            {
                outstandingWorkDict[workGuid] = dispatchedWorkRecord;
            }

            Job currentJob = GetJobFromJobsDict(simulationRecord.JobGuid);
            Simulation? simulation = currentJob.GetSimulation(simulationRecord.SimIdx);

            dynamic? innerSimulation = simulation!.InitSimResults(workGuid);

            Interlocked.Increment(ref DispatchedSimulationCount);
            return new SimulationDetails(workGuid, innerSimulation, RetryDelay: 0);
        }

        public SimulationDetails RequestCommuncationDelay()
        {
            return new SimulationDetails(Guid.Empty, default, RetryDelay: MagicNumbers.DefaultRetryDelaySeconds);
        }

        public SimulationDetails GetSimulation(Guid nodeGuid)
        {
            // check the pending queue
            if (TryGetPendingSimulation(out SimulationRecord? simulationRecord))
            {
                return DispatchSimulation(simulationRecord!, nodeGuid);
            }

            // check the retry queue
            if (TryGetRetrySimulation(out simulationRecord))
            {
                return DispatchSimulation(simulationRecord!, nodeGuid);
            }

            // update retry queue and check again
            if (IdentifyTimedOutOutstandingWork() && TryGetRetrySimulation(out simulationRecord))
            {
                return DispatchSimulation(simulationRecord!, nodeGuid);
            }

            // no simulations available right now
            return RequestCommuncationDelay();
        }

        #endregion // GET_A_SIMULATION

        #region WORK_FINISHED
        public bool WorkFinished(Guid workGuid)
        {
            if (!RemoveOutstandingWork(workGuid, out DispatchedWorkRecord? workRecord))
            {
                return false;
            }
            NodeInfo nodeInfo;
            lock(nodeInfoDictLock)
            {
                nodeInfo = nodeInfoDict[workRecord!.NodeGuid];
            }
            nodeInfo.FinishWork();

            SimulationRecord simRecord = workRecord!.SimRecord;
            SimulationResults? simResults = GetJobFromJobsDict(simRecord.JobGuid).GetSimulationResults(simRecord.SimIdx, workGuid);
            simResults!.ResultsStatus = ResultsStatuses.COMPLETE;

            Interlocked.Increment(ref TotalFinishedSimulationCount);
            return true;

        }
        #endregion // WORK_FINISHED

        #region ERRORED
        /// <summary>
        /// Old behavior: node logged error, and work was re-started later up to ErrorCountUntilFailure times (e.g. 10)
        ///  if a node is cancelled, work might never be marked as done, so we would like to auto-retry work. However if a node marks the work as errored, we should trust it by default and record the error as a failure and not re-run the work later.
        /// </summary>
        /// <param name="workGuid">Guid of work</param>
        /// <param name="errorDetails">Any extra info to log about the error</param>
        /// <returns>
        /// New behavior: logged errors are fine. Do not shutdown nodes.
        /// 
        /// Old behavior:
        /// Returns false:
        /// * To indicate that the worker node should early terminate.
        /// * Occurs if workGuid is invalid,
        /// * If worker node has logged too many errors.

        /// </returns>
        public bool RecordError(Guid workGuid, dynamic? errorDetails = default)
        {
            if (!RemoveOutstandingWork(workGuid, out DispatchedWorkRecord? workRecord))
            {
                return false;
            }
            NodeInfo nodeInfo;
            lock(nodeInfoDictLock)
            {
                nodeInfo = nodeInfoDict[workRecord!.NodeGuid];
            }
            int nodeErrorCnt = nodeInfo.ErrorWork();

            SimulationRecord simRecord = workRecord!.SimRecord;
            SimulationResults? simResults = GetJobFromJobsDict(simRecord.JobGuid).GetSimulationResults(simRecord.SimIdx, workGuid);
            simResults!.ResultsStatus = ResultsStatuses.FAILED;
            simResults.ErrorDetails = errorDetails;

            // Old behavior with error => retry not failure.
            // Old behavior => shutdown nodes after reporting errors.
            // if (simRecord.RetryCount < MagicNumbers.ErrorCountUntilFailure)
            // {
            //     retryQueue.Enqueue(new SimulationRecord(simRecord.JobGuid, simRecord.SimIdx, simRecord.RetryCount + 1));
            // }
            // else
            // {
            //     Interlocked.Increment(ref FailedSimulationCount);
            // }
            // 
            // return nodeErrorCnt < MagicNumbers.ErrorCountUntilFailure;
            

            // New: treat all logged errors as failures, and let nodes keep running.
            Interlocked.Increment(ref FailedSimulationCount);
            Interlocked.Increment(ref TotalSimulationErrorCount);
            return true;
        }
        #endregion // ERRORED

        #region RESULTS
        public bool AppendResults(Guid workGuid, dynamic incrementalResults)
        {
            DispatchedWorkRecord? workRecord;
            bool tryGetSuccess;
            lock(outstandingWorkDictLock)
            {
                tryGetSuccess = outstandingWorkDict.TryGetValue(workGuid, out workRecord);
            }

            if (!tryGetSuccess)
            {
                return false;
            }

            NodeInfo nodeInfo;
            lock(nodeInfoDictLock)
            {
                nodeInfo = nodeInfoDict[workRecord!.NodeGuid];
            }
            nodeInfo.UpdateWork();


            SimulationRecord simRecord = workRecord!.SimRecord;
            SimulationResults? simResults = GetJobFromJobsDict(simRecord.JobGuid).GetSimulationResults(simRecord.SimIdx, workGuid);
            simResults!.AppendResults(incrementalResults);

            Interlocked.Increment(ref TotalResultsReceivedCount);

            return true;

        }
        #endregion // RESULTS

        #region EXPORT_RESULTS
        public Job[] GetJobsArray()
        {
            lock(jobsDictLock)
            {
                return jobsDict.Select(kv => kv.Value).ToArray();
            }
        }
        public void ExportJobSummariesFile(string fileName, Job[] jobs)
        {
            ExportJobRecord[] jobSummaries = jobs.Select(x => x.ExportJobSummary()).ToArray();
            byte[] summaryBytes = JsonSerializer.SerializeToUtf8Bytes(jobSummaries);

            // using (var compressedFileStream = File.Create($"{fileName}.gz"))
            // using (var fileOut = new GZipStream(compressedFileStream, CompressionMode.Compress))
            using (var fileOut = File.Create($"{fileName}"))
            {
                fileOut.Write(summaryBytes, 0, summaryBytes.Length);
            }
        }

        public void ExportCompletedToFolder(string folder, bool compression = true)
        {
            Directory.CreateDirectory(folder);
            Job[] jobs = GetJobsArray();

            ExportJobSummariesFile($"{folder}/jobSummaries.json", jobs);

            Parallel.ForEach(jobs, job => {
                job.ExportCompletedSimulationsFolder(folder, compression: compression);
            });
        }

        public void ExportAllToFolder(string folder, bool compression = true)
        {
            Directory.CreateDirectory(folder);
            Job[] jobs = GetJobsArray();

            ExportJobSummariesFile($"{folder}/jobSummaries.json", jobs);

            Parallel.ForEach(jobs, job => {
                job.ExportAllSimulationsFolder(folder, compression: compression);
            });
        }

        public dynamic? ExportNodes()
        {
            lock (nodeInfoDictLock)
            {
                return nodeInfoDict.ToArray();
            }
        }
        #endregion // EXPORT_RESULTS

        #region STATS
        public dynamic? GetStats()
        {
            int outstandingSimulationCount, totalNodeCount, activeNodeCount;

            lock(outstandingWorkDictLock)
            {
                outstandingSimulationCount = outstandingWorkDict.Count;
            }

            lock(nodeInfoDictLock)
            {
                totalNodeCount = nodeInfoDict.Count;
                activeNodeCount = nodeInfoDict.Where(kv => kv.Value.IsNodeActive()).Count();
            }

            return new StatsRecord(TotalJobCount: GetPrimaryQueueCount(),
                                    DispatchedJobCount: CurrentJobIdx,
                                    TotalSimulationCount: TotalSimulationCount,
                                    DispatchedSimulationCount: DispatchedSimulationCount,
                                    OutstandingSimulationCount: outstandingSimulationCount,
                                    TotalSimulationErrorCount: TotalSimulationErrorCount,
                                    FailedSimulationCount:FailedSimulationCount,
                                    TotalResultsReceivedCount: TotalResultsReceivedCount,
                                    TotalFinishedSimulationCount: TotalFinishedSimulationCount,
                                    TotalNodeCount: totalNodeCount,
                                    SimulationsInRetryQueue: retryQueue.Count,
                                    ActiveNodeCount: activeNodeCount);
        }
        #endregion // STATS
    }

}

