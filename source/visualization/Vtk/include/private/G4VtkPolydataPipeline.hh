//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************

#ifndef G4VTKPOLYDATAPIPELINE_HH
#define G4VTKPOLYDATAPIPELINE_HH

#include "G4Point3D.hh"
#include "G4VVtkPipeline.hh"

#include <vtkSmartPointer.h>

#include <vector>

class G4String;
class G4VtkVisContext;
class G4Polyhedron;
class G4Polyline;

class vtkPoints;
class vtkCellArray;
class vtkPolyData;
class vtkPolyDataAlgorithm;
class vtkPolyDataMapper;
class vtkActor;

class G4VtkPolydataPipeline : public G4VVtkPipeline
{
  public:
    G4VtkPolydataPipeline(G4String name, const G4VtkVisContext& vc);
    ~G4VtkPolydataPipeline() override = default;

    void AddFilter(vtkSmartPointer<vtkPolyDataAlgorithm> f) { filters.push_back(f); }
    vtkSmartPointer<vtkPolyDataAlgorithm> GetFilter(G4int iFilter) { return filters[iFilter]; }
    G4int GetNumberOfFilters() { return (G4int)filters.size(); }
    vtkSmartPointer<vtkPolyDataAlgorithm> GetFinalFilter() { return filters[filters.size() - 1]; }

    void Enable() override;
    void Disable() override;

    void Print() override;
    void Modified() override;
    void Clear() override;

    virtual vtkSmartPointer<vtkActor> GetActor() { return actor; }

    virtual void SetPolydata(const G4Polyhedron& polyhedron);
    virtual void SetPolydata(const G4Polyline& polyline);
    virtual void SetPolydata(vtkPolyData *polydata);
    virtual void SetPolydataData(const G4Point3D& p);
    virtual void SetPolydataData(double x, double y, double z);

    virtual void SetActorTransform(G4double dx, G4double dy, G4double dz, G4double r00,
                                   G4double r01, G4double r02, G4double r10, G4double r11,
                                   G4double r12, G4double r20, G4double r21, G4double r22);
    virtual void SetActorColour(G4double r, G4double g, G4double b, G4double a);

    virtual vtkSmartPointer<vtkPolyData> GetPolydata() { return polydata; }

    virtual G4double* GetBounds() { return actor->GetBounds(); }

    static std::size_t MakeHash(const G4Polyhedron& p, const G4VtkVisContext& vc);

  protected:
    vtkSmartPointer<vtkPoints> polydataPoints;
    vtkSmartPointer<vtkCellArray> polydataCells;
    vtkSmartPointer<vtkPolyData> polydata;

    std::vector<vtkSmartPointer<vtkPolyDataAlgorithm>>
      filters;  // derived types can store filters in this vector
    vtkSmartPointer<vtkPolyDataMapper> mapper;  // polydagta mapper
    vtkSmartPointer<vtkActor> actor;  // all pipelines require an actor
};

#endif  // G4VTKPOLYDATAPIPELINE_HH
