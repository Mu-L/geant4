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
//
// ABLAXX statistical de-excitation model
// Jose Luis Rodriguez, UDC (translation from ABLA07 and contact person)
// Pekka Kaitaniemi, HIP (initial translation of ablav3p)
// Aleksandra Kelic, GSI (ABLA07 code)
// Davide Mancusi, CEA (contact person INCL)
// Aatos Heikkinen, HIP (project coordination)
//

#include "globals.hh"
#include <cmath>
#include <iostream>

#include "G4AblaInterface.hh"
#include "G4DoubleHyperDoubleNeutron.hh"
#include "G4DoubleHyperH4.hh"
#include "G4DynamicParticle.hh"
#include "G4ExcitationHandler.hh"
#include "G4HyperAlpha.hh"
#include "G4HyperH4.hh"
#include "G4HyperHe5.hh"
#include "G4HyperTriton.hh"
#include "G4IonTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysicalConstants.hh"
#include "G4PhysicsModelCatalog.hh"
#include "G4ReactionProduct.hh"
#include "G4ReactionProductVector.hh"
#include "G4SystemOfUnits.hh"

G4AblaInterface::G4AblaInterface(G4ExcitationHandler* ptr)
    : G4VPreCompoundModel(ptr, "ABLAXX")
    , ablaResult(new G4VarNtp)
    , theABLAModel(new G4Abla(ablaResult))
    , eventNumber(0)
    , secID(-1)
    , isInitialised(false)
{
    secID = G4PhysicsModelCatalog::GetModelID("model_" + GetModelName());
    // G4cout << "### NEW PrecompoundModel " << this << G4endl;
    if (!ptr)
        SetExcitationHandler(new G4ExcitationHandler);
    InitialiseModel();
    G4cout << G4endl << "G4AblaInterface::InitialiseModel() was right." << G4endl;
}

G4AblaInterface::~G4AblaInterface()
{
    applyYourselfResult.Clear();
    delete ablaResult;
    delete theABLAModel;
    delete GetExcitationHandler();
}

void G4AblaInterface::BuildPhysicsTable(const G4ParticleDefinition&) { InitialiseModel(); }

void G4AblaInterface::InitialiseModel()
{
    if (isInitialised)
        return;
    isInitialised = true;
    theABLAModel->initEvapora();
    theABLAModel->SetParameters();
    GetExcitationHandler()->Initialise();
}

G4HadFinalState* G4AblaInterface::ApplyYourself(const G4HadProjectile& thePrimary, G4Nucleus& theNucleus)
{
    // This method is adapted from  G4PreCompoundModel::ApplyYourself,
    // and it is used only by Binary Cascade (BIC) when the latter is coupled with
    // Abla for nuclear de-excitation. This method allows BIC+ABLA to be used also
    // for proton and neutron projectile with kinetic energies below 45 MeV, by
    // creating a "compound" nucleus made by the system "target nucleus +
    // projectile", before calling the DeExcite method.
    const G4ParticleDefinition* primary = thePrimary.GetDefinition();
    if (primary != G4Neutron::Definition() && primary != G4Proton::Definition())
    {
        G4ExceptionDescription ed;
        ed << "G4AblaModel is used for ";
        if (primary)
            ed << primary->GetParticleName();
        G4Exception("G4AblaInterface::ApplyYourself()", "had040", FatalException, ed, "");
        return nullptr;
    }

    G4int Zp = 0;
    G4int Ap = 1;
    if (primary == G4Proton::Definition())
        Zp = 1;
    G4double timePrimary = thePrimary.GetGlobalTime();
    G4int A = theNucleus.GetA_asInt();
    G4int Z = theNucleus.GetZ_asInt();
    G4LorentzVector p = thePrimary.Get4Momentum();
    G4double mass = G4NucleiProperties::GetNuclearMass(A, Z);
    p += G4LorentzVector(0.0, 0.0, 0.0, mass);

    G4Fragment anInitialState(A + Ap, Z + Zp, p);
    anInitialState.SetNumberOfExcitedParticle(1, Zp);
    anInitialState.SetNumberOfHoles(1, Zp);
    anInitialState.SetCreationTime(thePrimary.GetGlobalTime());
    anInitialState.SetCreatorModelID(secID);

    G4ReactionProductVector* deExciteResult = DeExcite(anInitialState);

    applyYourselfResult.Clear();
    applyYourselfResult.SetStatusChange(stopAndKill);
    for (auto const& prod : *deExciteResult)
    {
        G4DynamicParticle* aNewDP =
            new G4DynamicParticle(prod->GetDefinition(), prod->GetTotalEnergy(), prod->GetMomentum());
        G4HadSecondary aNew = G4HadSecondary(aNewDP);
        G4double time = std::max(prod->GetFormationTime(), 0.0);
        aNew.SetTime(timePrimary + time);
        aNew.SetCreatorModelID(prod->GetCreatorModelID());
        delete prod;
        applyYourselfResult.AddSecondary(aNew);
    }
    delete deExciteResult;
    return &applyYourselfResult;
}

G4ReactionProductVector* G4AblaInterface::DeExcite(G4Fragment& aFragment)
{
    if (!isInitialised)
        InitialiseModel();

    ablaResult->clear();

    const G4int ARem = aFragment.GetA_asInt();
    const G4int ZRem = aFragment.GetZ_asInt();
    const G4int SRem = -aFragment.GetNumberOfLambdas(); // Strangeness = - (Number of lambdas)
    const G4double eStarRem = aFragment.GetExcitationEnergy() / MeV;
    const G4double jRem = aFragment.GetAngularMomentum().mag() / hbar_Planck;
    const G4LorentzVector& pRem = aFragment.GetMomentum();
    const G4double pxRem = pRem.x() / MeV;
    const G4double pyRem = pRem.y() / MeV;
    const G4double pzRem = pRem.z() / MeV;

    ++eventNumber;

    theABLAModel->DeexcitationAblaxx(ARem, ZRem, eStarRem, jRem, pxRem, pyRem, pzRem, (G4int)eventNumber, SRem);

    G4ReactionProductVector* result = new G4ReactionProductVector;

    for (G4int j = 0; j < ablaResult->ntrack; ++j)
    { // Copy ABLA result to the EventInfo
        G4ReactionProduct* product = toG4Particle(ablaResult->avv[j],
                                                  ablaResult->zvv[j],
                                                  ablaResult->svv[j],
                                                  ablaResult->enerj[j],
                                                  ablaResult->pxlab[j],
                                                  ablaResult->pylab[j],
                                                  ablaResult->pzlab[j]);
        if (product)
        {
            product->SetCreatorModelID(secID);
            result->push_back(product);
        }
    }
    return result;
}

G4ParticleDefinition* G4AblaInterface::toG4ParticleDefinition(G4int A, G4int Z, G4int S) const
{
    if (A == 1 && Z == 1 && S == 0)
        return G4Proton::Proton();
    else if (A == 1 && Z == 0 && S == 0)
        return G4Neutron::Neutron();
    else if (A == 1 && Z == 0 && S == -1)
        return G4Lambda::Lambda();
    else if (A == -1 && Z == 1 && S == 0)
        return G4PionPlus::PionPlus();
    else if (A == -1 && Z == -1 && S == 0)
        return G4PionMinus::PionMinus();
    else if (A == -1 && Z == 0 && S == 0)
        return G4PionZero::PionZero();
    else if (A == 0 && Z == 0 && S == 0)
        return G4Gamma::Gamma();
    else if (A == 2 && Z == 1 && S == 0)
        return G4Deuteron::Deuteron();
    else if (A == 3 && Z == 1 && S == 0)
        return G4Triton::Triton();
    else if (A == 3 && Z == 2 && S == 0)
        return G4He3::He3();
    else if (A == 3 && Z == 1 && S == -1)
        return G4HyperTriton::Definition();
    else if (A == 4 && Z == 2 && S == 0)
        return G4Alpha::Alpha();
    else if (A == 4 && Z == 1 && S == -1)
        return G4HyperH4::Definition();
    else if (A == 4 && Z == 2 && S == -1)
        return G4HyperAlpha::Definition();
    else if (A == 4 && Z == 1 && S == -2)
        return G4DoubleHyperH4::Definition();
    else if (A == 4 && Z == 0 && S == -2)
        return G4DoubleHyperDoubleNeutron::Definition();
    else if (A == 5 && Z == 2 && S == -1)
        return G4HyperHe5::Definition();
    else if (A > 0 && Z > 0 && A > Z)
    { // Returns ground state ion definition.
        auto ionfromtable = G4IonTable::GetIonTable()->GetIon(Z, A, std::abs(S), 0); // S is the number of lambdas
        if (ionfromtable)
            return ionfromtable;
        else
        {
            G4cout << "Can't convert particle with A=" << A << ", Z=" << Z << ", S=" << S
                   << " to G4ParticleDefinition, trouble ahead" << G4endl;
            return 0;
        }
    }
    else
    { // Error, unrecognized particle
        G4cout << "Can't convert particle with A=" << A << ", Z=" << Z << ", S=" << S
               << " to G4ParticleDefinition, trouble ahead" << G4endl;
        return 0;
    }
}

G4ReactionProduct*
    G4AblaInterface::toG4Particle(G4int A, G4int Z, G4int S, G4double kinE, G4double px, G4double py, G4double pz) const
{
    G4ParticleDefinition* def = toG4ParticleDefinition(A, Z, S);
    if (def == 0)
    { // Check if we have a valid particle definition
        return 0;
    }

    const G4double energy = kinE * MeV;
    const G4ThreeVector momentum(px, py, pz);
    const G4ThreeVector momentumDirection = momentum.unit();
    G4DynamicParticle p(def, momentumDirection, energy);
    G4ReactionProduct* r = new G4ReactionProduct(def);
    (*r) = p;
    return r;
}

void G4AblaInterface::ModelDescription(std::ostream& outFile) const
{
    outFile << "ABLA++ does not provide an implementation of the ApplyYourself "
               "method!\n\n";
}

void G4AblaInterface::DeExciteModelDescription(std::ostream& outFile) const
{
    outFile << "ABLA++ is a statistical model for nuclear de-excitation. It simulates\n"
            << "the gamma emission and the evaporation of neutrons, light charged\n"
            << "particles and IMFs, as well as fission where applicable. The code\n"
            << "included in Geant4 is a C++ translation of the original Fortran\n"
            << "code ABLA07. Although the model has been recently extended to\n"
            << "hypernuclei by including the evaporation of lambda particles.\n"
            << "More details about the physics are available in the Geant4\n"
            << "Physics Reference Manual and in the reference articles.\n\n"
            << "References:\n"
            << "(1) A. Kelic, M. V. Ricciardi, and K. H. Schmidt, in Proceedings of Joint\n"
            << "ICTP-IAEA Advanced Workshop on Model Codes for Spallation Reactions,\n"
            << "ICTP Trieste, Italy, 4–8 February 2008, edited by D. Filges, S. "
               "Leray, Y. Yariv, A. Mengoni, A. Stanculescu, and G. Mank (IAEA "
               "INDC(NDS)-530, Vienna, 2008), pp. 181–221.\n\n"
            << "(2) J.L. Rodriguez-Sanchez, J.-C. David et al., Phys. Rev. C 98, 021602R (2018)\n"
            << "(3) J.L. Rodriguez-Sanchez et al., Phys. Rev. C 105, 014623 (2022)\n"
            << "(4) J.L. Rodriguez-Sanchez et al., Phys. Rev. Lett. 130, 132501 (2023)\n\n";
}
