using System;
using System.Linq;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.DependencyInjection;

using MinimalAPI;

var builder = WebApplication.CreateBuilder(args);
// builder.Host.ConfigureLogging(logging => logging.SetMinimumLevel(LogLevel.Warning));

builder.Services.AddSingleton<JobRepository>(new JobRepository());

builder.Services.AddControllers().AddJsonOptions(x =>
{
    x.JsonSerializerOptions.Converters.Add(new JsonStringEnumConverter());
});

var app = builder.Build();

app.MapPost("/register", (JobRepository repo, dynamic? nodeDetails) =>
{
    return Results.Ok(repo.RegisterNode(nodeDetails: nodeDetails));
});

app.MapPost("/shutdownnode/{nodeGuid}", (JobRepository repo, Guid nodeGuid) =>
{
    repo.ShutdownNode(nodeGuid);
    return Results.Ok();
});

app.MapGet("/stats", (JobRepository repo) =>
{
    return Results.Ok(repo.GetStats());
});

app.MapGet("/exportnodes", (JobRepository repo) =>
{
    return Results.Ok(repo.ExportNodes());
});

app.MapPost("/exportcompletedfolder", (JobRepository repo, ExportFolderName folderInfo) =>
{
    repo.ExportCompletedToFolder(folderInfo.folderName, compression: folderInfo.compression);
    return Results.Ok();
});

app.MapPost("/exportallfolder", (JobRepository repo, ExportFolderName folderInfo) =>
{
    repo.ExportAllToFolder(folderInfo.folderName, compression: folderInfo.compression);
    return Results.Ok();
});
/*
app.MapPost("/exportresultsfilequick", (JobRepository repo, ExportFileName fileInfo) =>
{
    repo.ExportToFileQuick(fileInfo.fileName);
    return Results.Ok();
});

app.MapGet("/exportresultsquick", (JobRepository repo) =>
{
    return Results.Ok(repo.ExportResultsQuick());
});

app.MapGet("/exportresults", (JobRepository repo) =>
{
    return Results.Ok(repo.ExportResults());
});
*/

app.MapPost("/submitjob", (JobRepository repo, JobSubmissionRecord job) =>
{
    return Results.Ok(repo.EnqueueJob(job.InnerSimulations, job.JobDetails));
});

//app.MapPost("/submitjob", (JobRepository repo, dynamic[] innerSimulations) =>
//{
//    return Results.Ok(repo.EnqueueJob(innerSimulations));
//});

app.MapGet("/simulation/{nodeGuid}", (JobRepository repo, Guid nodeGuid) =>
{
    return Results.Ok(repo.GetSimulation(nodeGuid));
});

app.MapPost("/results/{workGuid}", (JobRepository repo, Guid workGuid, dynamic incrementalResults) =>
{
    return Results.Ok(repo.AppendResults(workGuid, incrementalResults));
});

app.MapPost("/finished/{workGuid}", (JobRepository repo, Guid workGuid) =>
{
    return Results.Ok(repo.WorkFinished(workGuid));
});

app.MapPost("/errors/{workGuid}", (JobRepository repo, Guid workGuid, dynamic? errorDetails) =>
{
    return Results.Ok(repo.RecordError(workGuid, errorDetails));
});

app.Run();
