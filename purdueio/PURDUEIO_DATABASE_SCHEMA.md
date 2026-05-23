# Purdue.io Database Schema Reference

This document captures the **exact Purdue.io relational schema** currently used by the source in this workspace.

## Source Of Truth
- `src/Database/ApplicationDbContext.cs`
- `src/Database.Migrations.Npgsql/Migrations/20231112213223_Initial.cs`
- `src/Api/EdmModelBuilder.cs`

Resolved against local path:
- `/Users/winstonlin/Downloads/work/ai-academic-advisor/purdueio/PurdueApi`

## Schema Version
- EF Core migration: `20231112213223_Initial` (Npgsql)
- Database objects are created in the `public` schema by default.

## Tables (PostgreSQL Types)

### `Campuses`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `Code` `character varying(12)` NULL
- `Name` `text` NULL
- `ZipCode` `character varying(5)` NULL

Indexes:
- `IX_Campuses_Code` UNIQUE (`Code`)
- `IX_Campuses_Name` UNIQUE (`Name`)

---

### `Buildings`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `CampusId` `uuid` NOT NULL
- `Name` `text` NULL
- `ShortCode` `text` NULL

Foreign keys:
- `FK_Buildings_Campuses_CampusId`:
  `CampusId -> Campuses(Id)` ON DELETE CASCADE

Indexes:
- `IX_Buildings_CampusId_ShortCode` UNIQUE (`CampusId`, `ShortCode`)
- `IX_Buildings_Name` (`Name`)
- `IX_Buildings_ShortCode` (`ShortCode`)

---

### `Rooms`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `Number` `text` NULL
- `BuildingId` `uuid` NOT NULL

Foreign keys:
- `FK_Rooms_Buildings_BuildingId`:
  `BuildingId -> Buildings(Id)` ON DELETE CASCADE

Indexes:
- `IX_Rooms_BuildingId` (`BuildingId`)
- `IX_Rooms_Number` (`Number`)

---

### `Subjects`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `Name` `text` NULL
- `Abbreviation` `character varying(6)` NULL

Indexes:
- `IX_Subjects_Abbreviation` UNIQUE (`Abbreviation`)
- `IX_Subjects_Name` (`Name`)

---

### `Courses`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `Number` `character varying(16)` NULL
- `SubjectId` `uuid` NOT NULL
- `Title` `text` NULL
- `CreditHours` `double precision` NOT NULL
- `Description` `text` NULL

Foreign keys:
- `FK_Courses_Subjects_SubjectId`:
  `SubjectId -> Subjects(Id)` ON DELETE CASCADE

Indexes:
- `IX_Courses_Number` (`Number`)
- `IX_Courses_SubjectId` (`SubjectId`)
- `IX_Courses_Title` (`Title`)

---

### `Terms`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `Code` `character varying(16)` NULL
- `Name` `text` NULL
- `StartDate` `date` NULL
- `EndDate` `date` NULL

Indexes:
- `IX_Terms_Code` UNIQUE (`Code`)
- `IX_Terms_Name` UNIQUE (`Name`)

---

### `Classes`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `CourseId` `uuid` NOT NULL
- `TermId` `uuid` NOT NULL
- `CampusId` `uuid` NOT NULL

Foreign keys:
- `FK_Classes_Courses_CourseId`:
  `CourseId -> Courses(Id)` ON DELETE CASCADE
- `FK_Classes_Terms_TermId`:
  `TermId -> Terms(Id)` ON DELETE CASCADE
- `FK_Classes_Campuses_CampusId`:
  `CampusId -> Campuses(Id)` ON DELETE CASCADE

Indexes:
- `IX_Classes_CourseId` (`CourseId`)
- `IX_Classes_TermId` (`TermId`)
- `IX_Classes_CampusId` (`CampusId`)

---

### `Sections`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `Crn` `character varying(16)` NULL
- `ClassId` `uuid` NOT NULL
- `Type` `text` NULL
- `StartDate` `date` NULL
- `EndDate` `date` NULL

Foreign keys:
- `FK_Sections_Classes_ClassId`:
  `ClassId -> Classes(Id)` ON DELETE CASCADE

Indexes:
- `IX_Sections_ClassId` (`ClassId`)
- `IX_Sections_Crn` (`Crn`)

---

### `Meetings`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `SectionId` `uuid` NOT NULL
- `Type` `text` NULL
- `StartDate` `date` NULL
- `EndDate` `date` NULL
- `DaysOfWeek` `smallint` NOT NULL
- `StartTime` `time without time zone` NULL
- `Duration` `interval` NOT NULL
- `RoomId` `uuid` NULL

Foreign keys:
- `FK_Meetings_Sections_SectionId`:
  `SectionId -> Sections(Id)` ON DELETE CASCADE
- `FK_Meetings_Rooms_RoomId`:
  `RoomId -> Rooms(Id)` (no cascade configured)

Indexes:
- `IX_Meetings_SectionId` (`SectionId`)
- `IX_Meetings_RoomId` (`RoomId`)

---

### `Instructors`
- `Id` `uuid` NOT NULL PRIMARY KEY
- `Name` `text` NULL
- `Email` `character varying(254)` NULL

Indexes:
- `IX_Instructors_Name` (`Name`)
- `IX_Instructors_Email` (`Email`)

---

### `MeetingInstructor` (join table)
- `MeetingId` `uuid` NOT NULL
- `InstructorId` `uuid` NOT NULL
- PRIMARY KEY (`MeetingId`, `InstructorId`)

Foreign keys:
- `FK_MeetingInstructor_Meetings_MeetingId`:
  `MeetingId -> Meetings(Id)` ON DELETE CASCADE
- `FK_MeetingInstructor_Instructors_InstructorId`:
  `InstructorId -> Instructors(Id)` ON DELETE CASCADE

Indexes:
- `IX_MeetingInstructor_InstructorId` (`InstructorId`)

## Relationship Graph (Join Keys)
- `Courses.SubjectId -> Subjects.Id`
- `Classes.CourseId -> Courses.Id`
- `Classes.TermId -> Terms.Id`
- `Classes.CampusId -> Campuses.Id`
- `Sections.ClassId -> Classes.Id`
- `Meetings.SectionId -> Sections.Id`
- `Meetings.RoomId -> Rooms.Id`
- `Rooms.BuildingId -> Buildings.Id`
- `Buildings.CampusId -> Campuses.Id`
- `MeetingInstructor.MeetingId -> Meetings.Id`
- `MeetingInstructor.InstructorId -> Instructors.Id`

## OData Entity Set Mapping
The API exposes these entity sets under `/odata`:
- `/odata/Campuses`
- `/odata/Buildings`
- `/odata/Rooms`
- `/odata/Terms`
- `/odata/Courses`
- `/odata/Classes`
- `/odata/Sections`
- `/odata/Meetings`
- `/odata/Subjects`
- `/odata/Instructors`

## Notes For Future Extension
- Purdue.io schema currently has **no tables for degree programs or degree requirements**.
- To keep compatibility, add new requirement tables with FKs to existing IDs (especially `Courses.Id`, optionally `Terms.Id` and `Campuses.Id`).
- `DaysOfWeek` in `Meetings` is a bit-flag enum stored as `smallint`.

